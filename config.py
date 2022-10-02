CHECKPOINT = "vinai/phobert-base"
DATA_PATH = "uit-nlp/vietnamese_students_feedback"

EPOCHS = 3
VAL_EACH_EPOCH = 2
NUM_CLASSES = 3
BATCH_SIZE = 64
MID_HIDDEN_LAYER = 768
THRESHOLD=0.5

LOGGER = {
  "name": "PhoBERT_base",
  "version": 0,
}

EVALUATE_WHILE_TRAINING = True
CKPT_DIR = "./checkpoint"
CKPT_PATH = None   # Checkpoint for inference, default is None
CKPT_RESUME_TRAIN = None    # Checkpoint for continue training, default is None