import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DICT = {
    0: "FeedForward-base",
    1: "FeedForward-large",
    2: "LSTM-base",
    3: "LSTM-large"
}

MODEL1 = MODEL_DICT[0]
MODEL2 = MODEL_DICT[1]

CKPT1 = ""
CKPT2 = ""

