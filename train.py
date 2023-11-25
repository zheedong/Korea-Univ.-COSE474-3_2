# BERT training from transformers pretrained
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

import pytorch_lightning as pl

class BartFineTuner(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=5e-5)

model.save_pretrained('./finetuned_bart')
tokenizer.save_pretrained('./finetuned_bart')

def main():
    # Your code for loading data, initializing the dataset, model, tokenizer, and trainer
    # Train the model
    # Save the model and tokenizer
    trainer = pl.Trainer(gpus=1, max_epochs=3, progress_bar_refresh_rate=30)
    trainer.fit(BartFineTuner(model), train_loader)


if __name__ == "__main__":
    main()
