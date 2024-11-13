from datasets import load_dataset
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# Load the CLINC150 dataset
def load_data():
    dataset = load_dataset("clinc_oos", "plus")
    return dataset

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
MAX_TOKEN_LENGTH = 128

class CLINC150Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        intent = self.data[idx]['intent']
        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length, padding='max_length',
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(intent, dtype=torch.long)
        }

def create_data_loader(data, tokenizer, batch_size=32, max_length=128):
    ds = CLINC150Dataset(data, tokenizer, max_length=max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
