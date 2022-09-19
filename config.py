import torch

CHECKPOINT = "vinai/phobert-base"
DATA_PATH = "uit-nlp/vietnamese_students_feedback"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WARM_UP_EPOCH = 3
EPOCH = 10
LOG_INTERVAL= 50
NUM_CLASSES = 3
BATCH_SIZE = 64
MID_HIDDEN_LAYER = 384

EVALUATE_WHILE_TRAINING = True