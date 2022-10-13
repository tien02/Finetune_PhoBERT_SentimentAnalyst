MODE = "TRAIN"  # 'TRAIN' to train model, ensemble to use ensemble
CHECKPOINT = "vinai/phobert-base"
DATA_PATH = "uit-nlp/vietnamese_students_feedback"

EPOCHS = 3
VAL_EACH_EPOCH = 2
NUM_CLASSES = 3
BATCH_SIZE = 64
THRESHOLD=0.5

MODEL = "FeedForward-base"   # "FeedForward"/"LSTM" + '-' +  'base'/'large'

LOGGER = {
  "name": "PhoBERT_base",
  "version": 0,
}

CKPT_DIR = "./checkpoint"
CKPT_PATH = None   # Checkpoint for inference, default is None
CKPT_RESUME_TRAIN = None    # Checkpoint for continue training, default is None