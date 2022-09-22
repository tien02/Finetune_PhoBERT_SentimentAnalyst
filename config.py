import torch

CHECKPOINT = "vinai/phobert-base"
DATA_PATH = "uit-nlp/vietnamese_students_feedback"

EPOCHS = 24
VAL_EACH_EPOCH = 4
NUM_CLASSES = 3
BATCH_SIZE = 64
MID_HIDDEN_LAYER = 384
THRESHOLD=0.5

EVALUATE_WHILE_TRAINING = True
CKPT_DIR = "./checkpoint"
CKPT_PATH = None    # Checkpoint for inference, default is None
CKPT_RESUME_TRAIN = None    # Checkpoint for continue training, default is None