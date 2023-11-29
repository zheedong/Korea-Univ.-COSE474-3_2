import torch
import torch.nn as nn
import torch.distributed as dist_fn
from einops import rearrange
from typing import List, Tuple, Optional
from torch.nn import functional as F

from vqgan.diffusionmodules import Encoder, Decoder


class VectorQuantizer(nn.Module):
    """
    Modified version of VectorQuantizer in the original VQGAN repository,
    removing unncessary modules for sampling and making the variable names and
    formats consistent to our VQ-VAE
    """
    def __init__(self, dim, n_embed, beta):
        super().__init__()
        self.n_embed = n_embed # 16384
        self.dim = dim # 256
        self.beta = beta

        self.embedding = nn.Embedding(self.n_embed, self.dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        return z_q, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class EMAVectorQuantizer(nn.Module):
    """
    EMAVectorQuantizer
    """
    def __init__(self,
                 dim: int,
                 n_embed: int,
                 beta: float,
                 decay: float = 0.99,
                 eps: float = 1e-5) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.dim = dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        embedding = torch.randn(n_embed, dim)
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(self.n_embed))
        self.register_buffer("embedding_avg", embedding.clone())

    def forward(self,
                z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z = rearrange(z, 'b c h w -> b h w c').contiguous()  # [B,C,H,W] -> [B,H,W,C]
        z_flattened = z.view(-1, self.dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, self.embedding).view(z.shape)
        embed_onehot = F.one_hot(min_encoding_indices, self.n_embed).type(z_flattened.dtype)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = embed_onehot.transpose(0, 1) @ z_flattened

            dist_fn.all_reduce(embed_onehot_sum, op=dist_fn.ReduceOp.SUM)
            dist_fn.all_reduce(embed_sum, op=dist_fn.ReduceOp.SUM)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embedding_avg / cluster_size.unsqueeze(1)

            self.embedding.data.copy_(embed_normalized)

        diff = self.beta * torch.mean((z_q.detach() - z) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, min_encoding_indices

    def get_codebook_entry(self,
                           indices: torch.LongTensor,
                           shape: Optional[List[int]] = None) -> torch.FloatTensor:
        z_q = F.embedding(indices, self.embedding)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class VQGAN(nn.Module):
    def __init__(self, n_embed, embed_dim, ema_update, hparams):
        super().__init__()
        self.encoder = Encoder(**hparams)
        self.decoder = Decoder(**hparams)

        if ema_update:
            self.quantize = EMAVectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)
        else:
            self.quantize = VectorQuantizer(dim=embed_dim, n_embed=n_embed, beta=0.25)
        self.quant_conv = torch.nn.Conv2d(hparams.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, hparams.z_channels, 1)
        self.latent_dim = hparams.attn_resolutions[0]

    def forward(self, x):
        quant = self.encode(x)
        dec = self.decode(quant)
        return dec

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant = self.quantize(h)[0]
        quant = rearrange(quant, 'b h w c -> b c h w').contiguous()
        return quant

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code):
        quant = self.quantize.get_codebook_entry(code)
        quant = quant.permute(0, 3, 1, 2)
        dec = self.decode(quant)
        return dec

    def get_recon_imgs(self, x_real, x_recon):
        x_real = x_real * 0.5 + 0.5
        x_recon = x_recon * 0.5 + 0.5
        x_recon = torch.clamp(x_recon, 0, 1)
        return x_real, x_recon

    def get_codes(self, x, timing=None, latent_dim=None):
        if timing is not None:
            with timing.timeit('stage1.encoder'):
                h = self.encoder(x, timing) # 4 x 256 x 16 x 16
            with timing.timeit('stage1.vec_quantizer'):
                h = self.quant_conv(h)
                codes = self.quantize(h)[1].view(x.shape[0],
                                                 self.latent_dim * self.latent_dim if latent_dim is None
                                                 else 32 * 32)
        else:
            h = self.encoder(x, timing)  # 4 x 256 x 16 x 16
            h = self.quant_conv(h)
            codes = self.quantize(h)[1].view(x.shape[0],
                                             self.latent_dim * self.latent_dim if latent_dim is None
                                             else 32 * 32)
        return codes
