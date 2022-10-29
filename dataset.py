import torch
import torch.nn as nn
import pandas as pd
import config.train_config as train_config
from torch.utils.data import Dataset
from transformers import PhobertTokenizer
from utils import preprocess_fn

tokenizer = PhobertTokenizer.from_pretrained(train_config.CHECKPOINT)

class UIT_VFSC_Dataset(Dataset):
    def __init__(self, root_dir, label="sentiments"):
        self.dataframe = pd.read_csv(root_dir, sep="\t")
        self.label = label
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        df = self.dataframe.iloc[index]
        X = df["sents"]
        y = df[self.label]

        x_tokens = preprocess_fn(X)
        tokens = tokenizer(x_tokens)

        return torch.tensor(tokens["input_ids"]), torch.tensor(tokens["attention_mask"]), torch.tensor(y), len(tokens["attention_mask"])
        
def collate_fn(batch):
    input_ids_list, attn_mask_list, label_list, length_list = [], [], [], []
    for input_ids, attn_mask, label, length in batch:
        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)
        label_list.append(label)
        length_list.append(length)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    length_list = torch.tensor(length_list)
    
    input_ids_list = nn.utils.rnn.pad_sequence(input_ids_list)
    attn_mask_list = nn.utils.rnn.pad_sequence(attn_mask_list)

    return input_ids_list, attn_mask_list, label_list