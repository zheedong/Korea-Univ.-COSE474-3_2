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
import numpy as np
import os
import glob

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
        pdb.set_trace()
        # [b, 256] image tokens
        token_batch_img = self.model_vq.get_codes(batch_img).to(self.device)
        token_batch_img += self.original_num_tokens

        # Process each item in the batch
        input_ids = []
        attention_masks = []
        labels = []
        for batch_idx, gt_txt in enumerate(batch_gt_txt):
            # Encode text
            text_tokens = self.tokenizer.encode(gt_txt, add_special_tokens=False)

            # Concatenate text tokens with image tokens
            combined_tokens = text_tokens + token_batch_img[batch_idx].tolist()

            # Create attention mask (1s for text tokens, 0s for image tokens to be predicted)
            attn_mask = [1] * len(text_tokens) + [0] * len(token_batch_img[batch_idx])

            # Prepare labels (ignore text tokens by setting them to -100)
            label = [-100] * len(text_tokens) + token_batch_img[batch_idx].tolist()

            input_ids.append(combined_tokens)
            attention_masks.append(attn_mask)
            labels.append(label)
        # Pad sequences to max length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        attention_masks = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in attention_masks], batch_first=True, padding_value=0).to(self.device)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbl) for lbl in labels], batch_first=True, padding_value=-100).to(self.device)

        # Forward pass through the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch.img, batch.gt_txt[0])
        loss = outputs.loss
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch.img, batch.gt_txt[0])
        loss = outputs.loss
        # Log validation loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        # Using StepLR as an example, adjust parameters as needed
        scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

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

    # bad words ids for forced decoding to generate image
    text_vocab_size = config.vocab_size
    bad_words_ids = [[i] for i in range(text_vocab_size)]
    cfg.original_num_tokens = config.vocab_size
    pdb.set_trace()
    config.vocab_size = config.vocab_size + vqgan_n_embed

    ckpt_root = './lightning_logs/version_76/checkpoints/'
    # Fine model_path which is end with .ckpt in ckpt_root
    model_path = glob.glob(os.path.join(ckpt_root, '*.ckpt'))[0]

    model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
    model.eval()

    vqgan_t5 = CustomT5Model(cfg, model, tokenizer)

    input_ids = tokenizer("A black Honda motorcycle parked in front of a garage.", return_tensors="pt").input_ids
    with torch.no_grad():
        # outputs = model.generate(input_ids)
        outputs = model.generate(input_ids,
                                do_sample=True,
                                min_length=256,
                                max_new_tokens=256,
                                num_beams=4,
                                temperature=0.9,
                                top_p=0.9,
                                repetition_penalty=0.8,
                                no_repeat_ngram_size=1,
                                bad_words_ids=bad_words_ids,
                                )
        print(outputs)
        outputs = outputs[:, 1:] - text_vocab_size
        print(outputs)
        pixels = vqgan_t5.image_generate(outputs)
        vqgan_t5.save_image(pixels, f'test2.jpg')