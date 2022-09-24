import config
import torch
from pyvi import ViTokenizer
from dataset import tokenizer
from trainer import PhoBERTTrainer
from termcolor import colored

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PhoBERTTrainer()
model.to(device)
checkpoint = torch.load(config.CKPT_PATH, map_location=device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.freeze()

map_dict = {
    0: "Negative",
    1: "Neural",
    2: "Positive"
}

if __name__ == "__main__":
    print("Enter -1 to exit...")
    while True:
        sentence = input("Enter a sentence: ")
        if sentence == "-1":    break
        seg_sentence = ViTokenizer.tokenize(sentence.lower())
        tokens = tokenizer(seg_sentence, return_tensors='pt')
        with torch.no_grad():
            output = model(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
            pred_label = int(torch.argmax(output, dim=1).item())
            accuracy = round(output[0][pred_label].item(), 3)
            if pred_label == 2:
                print("\tResult: "+ colored(map_dict[pred_label], "blue") + ", Accuracy: " + colored(accuracy, "blue") + "\n")
            elif pred_label == 1:
                print("\tResult: "+ colored(map_dict[pred_label], "yellow") + ", Accuracy: " + colored(accuracy, "yellow") + "\n")
            else:
                print("\tResult: "+ colored(map_dict[pred_label], "red") + ", Accuracy: " + colored(accuracy, "red") + "\n")
    print(colored("Done...", "green"))