import os
import torch
import lightning as pl

from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Config
from datamodules import build_datamodule, build_transform
from visual_bart.utils.config import build_config
from vqgan.vqgan import VQGAN
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pdb
from PIL import Image

def resize_tok_embeddings(model, new_voca_size, text_vocab_size):
    """_summary_

    Args:
        model (_type_): _description_
        new_voca_size (_type_): _description_
    """
    # Increase shared
    old_tok_embeddings = model.shared
    new_tok_embeddings = torch.nn.Embedding(
        new_voca_size, old_tok_embeddings.weight.shape[1]
    )

    new_tok_embeddings.weight.data[:text_vocab_size] = old_tok_embeddings.weight.data
    model.shared = new_tok_embeddings

    # Increase encoder
    old_tok_embeddings = model.encoder.embed_tokens
    new_tok_embeddings = torch.nn.Embedding(
        new_voca_size, old_tok_embeddings.weight.shape[1]
    )

    new_tok_embeddings.weight.data[:text_vocab_size] = old_tok_embeddings.weight.data
    model.encoder.embed_tokens = new_tok_embeddings

    # Increase Decoder
    old_tok_embeddings = model.decoder.embed_tokens
    new_tok_embeddings = torch.nn.Embedding(
        new_voca_size, old_tok_embeddings.weight.shape[1]
    )

    new_tok_embeddings.weight.data[:text_vocab_size] = old_tok_embeddings.weight.data
    model.decoder.embed_tokens = new_tok_embeddings

    # Increase lm head
    old_lm_head = model.lm_head
    new_lm_head = torch.nn.Linear(
        old_lm_head.weight.shape[1], new_voca_size
    )

    new_lm_head.weight.data[:text_vocab_size] = old_lm_head.weight.data
    model.lm_head = new_lm_head

class CustomT5Model(pl.LightningModule):
    def __init__(self, cfg, model, tokenizer):
        super().__init__()
        self.model = model
        self.model_vq = VQGAN(
           n_embed=cfg.vqgan.n_embed,
           embed_dim=cfg.vqgan.embed_dim,
           ema_update=False,
           hparams=cfg.vqgan.hparams,
        )
        self.model_vq = self.load_vqgan(self.model_vq, cfg.vqgan.path).to(self.device).eval()
        self.tokenizer = tokenizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.original_num_tokens = cfg.original_num_tokens
        self.vqgan_codebook_size = cfg.vqgan.n_embed

    def load_vqgan(self, model_vqvae, ckpt_path):
        if isinstance(model_vqvae, VQGAN):
            vqgan_ckpt = torch.load(ckpt_path,
                                    map_location=self.device)['state_dict']
            vqgan_ckpt_ = {}
            for key, value in vqgan_ckpt.items():
                if not key.startswith('loss') and not key.startswith('discriminator'):
                    if not key.startswith('generator'):
                        vqgan_ckpt_[key] = value
                    else:
                        vqgan_ckpt_[key[10:]] = value
            model_vqvae.load_state_dict(vqgan_ckpt_)
        else:
            vqvae_ckpt = torch.load(ckpt_path, map_location=self.device)['state_dict']
            model_vqvae.load_state_dict(vqvae_ckpt)
        return model_vqvae
    
    def image_generate(self, vqgan_token):

        zshape = (1, 256, 16, 16)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])

        quant = self.model_vq.quantize.get_codebook_entry(vqgan_token, shape=bhwc)
        dec = self.model_vq.decode(quant)

        return dec[0]

    def save_image(self, x, path):
        c,h,w = x.shape
        assert c==3
        x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
        Image.fromarray(x).save(path)

    def forward(self, batch_img, batch_gt_txt):
        # [b, 256] image tokens
        labels = self.model_vq.get_codes(batch_img).to(self.device)
        labels += self.original_num_tokens

        # Process each item in the batch
        input_ids = []
        for batch_idx, gt_txt in enumerate(batch_gt_txt):
            # Encode text
            text_tokens = self.tokenizer(gt_txt, add_special_tokens=False, return_tensors="pt").input_ids
            # Add padding
            text_tokens = torch.nn.functional.pad(text_tokens, (0, 128 - text_tokens.shape[1]), mode='constant', value=0)
            input_ids.append(text_tokens)

        # Stack the list as a tensor
        input_ids = torch.stack(input_ids, dim=0).squeeze(1).to(self.device)

        # Forward pass through the model
        outputs = self.model(input_ids=input_ids, labels=labels)

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch.img, [" ".join(text) for text in batch.gt_txt])
        loss = outputs.loss
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return loss

    def validation_step(self, batch, batch_idx):
        # Each gt has 5.
        outputs = self.forward(batch.img, [" ".join(text) for text in batch.gt_txt])
        loss = outputs.loss
        # Log validation loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # Using StepLR as an example, adjust parameters as needed
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.9)

        # If you want to use ReduceLROnPlateau
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Change to the metric you want to track
            },
        }


if __name__ == "__main__":
    cfg, cfg_yaml = build_config()

    cfg.dataset.eval_center_crop = True

    cfg.dataset.type = 'mapstyle'
    cfg.dataset.gt_text = True

    vqgan_n_embed = cfg.vqgan.n_embed

    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    config = T5Config.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    original_num_tokens = config.vocab_size
    cfg.original_num_tokens = original_num_tokens
    resize_tok_embeddings(model, original_num_tokens + vqgan_n_embed, original_num_tokens)


    val_transform = build_transform(
        cfg=cfg,
        split="val",
    )

    datamodule = build_datamodule(
        cfg=cfg,
        train_transform=val_transform,
        val_transform=val_transform,
        pin_memory=False,
        epoch=0,
        total_gpus=cfg.dist.n_gpus,
    )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()

    # Overfitting test
    #train_dataloader = datamodule.overfit_dataloader()

    trainer = pl.Trainer(max_epochs=100, accelerator='gpu', log_every_n_steps=1, check_val_every_n_epoch=1)
    t5_model = CustomT5Model(cfg, model, tokenizer)
    trainer.fit(t5_model, train_dataloader, val_dataloader)
