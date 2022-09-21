import torch

CHECKPOINT = "vinai/phobert-base"
DATA_PATH = "uit-nlp/vietnamese_students_feedback"

EPOCHS = 10
VAL_EACH_EPOCH = 2
NUM_CLASSES = 3
BATCH_SIZE = 64
MID_HIDDEN_LAYER = 384
THRESHOLD=0.5

EVALUATE_WHILE_TRAINING = True