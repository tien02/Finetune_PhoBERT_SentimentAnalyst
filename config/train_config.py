CHECKPOINT = "vinai/phobert-base"
DATA_PATH = "/Users/tiendang/Documents/Work/Project/Research/nlp/cs221_sentanalyst/uit_vsfc_data/train.csv"

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
CKPT_PATH =  "checkpoint/epoch=23-step=4296.ckpt"  # Checkpoint for inference, default is None
CKPT_RESUME_TRAIN = None    # Checkpoint for continue training, default is None