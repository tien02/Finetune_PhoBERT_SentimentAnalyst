import config
import torch
from pyvi import ViTokenizer
from dataset import tokenizer
from trainer import PhoBERTTrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PhoBERTTrainer()
model.to(device)
checkpoint = torch.load(config.CKPT_PATH)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.freeze()

map_dict = {
    0: "Negative",
    1: "Neural",
    2: "Positive"
}

if __name__ == "__main__":
    sentence = input("Enter a sentence: ")
    seg_sentence = ViTokenizer.tokenize(sentence)
    tokens = tokenizer(seg_sentence, return_tensors='pt')

    with torch.no_grad():
        output = model(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
        print(f">> RESULT: {map_dict.get(int(torch.argmax(output, dim=1), -1))}")
    print("Done...")