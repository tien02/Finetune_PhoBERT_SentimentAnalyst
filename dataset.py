import config.train_config as train_config
from torch.utils.data import DataLoader
from transformers import PhobertTokenizer, DataCollatorWithPadding
from datasets import load_dataset

tokenizer = PhobertTokenizer.from_pretrained(train_config.CHECKPOINT)

def tokenize_function(sentence):
    return tokenizer(sentence['sentence'], truncation=True)

dataset = load_dataset(train_config.DATA_PATH)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(['sentence', 'topic', 'token_type_ids'])

train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=train_config.BATCH_SIZE, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=train_config.BATCH_SIZE, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=train_config.BATCH_SIZE, collate_fn=data_collator)