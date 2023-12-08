import os
import torch
import lightning as pl
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from datamodules import build_datamodule, build_transform
from visual_bart.utils.config import build_config
from vqgan.vqgan import VQGAN
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import pdb

class CustomBartModel(pl.LightningModule):
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

    def forward(self, batch_img, batch_gt_txt):
        # [b, 256] image tokens
        token_batch_img = self.model_vq.get_codes(batch_img).to(self.device)
        token_batch_img += original_num_tokens

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

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    config = BartConfig.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large', config=config)

    original_num_tokens = model.lm_head.out_features
    model.shared = torch.nn.Embedding(original_num_tokens + vqgan_n_embed, config.d_model, padding_idx=1)
    model.resize_token_embeddings(original_num_tokens + vqgan_n_embed)


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
    bart_model = CustomBartModel(cfg, model, tokenizer)
    trainer.fit(bart_model, train_dataloader, val_dataloader)
