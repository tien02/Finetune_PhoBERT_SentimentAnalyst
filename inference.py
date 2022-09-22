import config
import torch
from pyvi import ViTokenizer
from dataset import tokenizer
from model import PhoBertClassifier

model = PhoBertClassifier()
checkpoint = torch.load(config.CKPT_PATH)
model.load_state_dict(checkpoint)
model.eval()
model.freeze()

if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    seg_sentence = ViTokenizer.tokenize(sentence)
    tokens = tokenizer(seg_sentence)

    with torch.no_grad():
        output = model(tokens["input_ids"], tokens["attn_mask"])
        print(output)