import pytorch_lightning as pl
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample data
texts = ["Hello, world!", "BERT is awesome!"]
labels = [0, 1]  # Example labels

# Tokenize the data
encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

# Create a PyTorch dataset
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = CustomDataset(encodings, labels)

# Define PyTorch Lightning model
class BertClassifier(pl.LightningModule):
    def __init__(self, num_labels=2):
        super(BertClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        total_steps = len(self.train_dataloader()) * self.trainer.max_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update the scheduler every step
                'frequency': 1,  # Frequency of scheduler updates
            }
        }

# Initialize the model
model = BertClassifier()

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=1, gpus=0)  # Set gpus=1 if you have a GPU

# Train the model
trainer.fit(model, train_loader)
