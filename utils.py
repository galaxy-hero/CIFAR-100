import torch

LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 32
NUM_EPOCHS = 30
EARLY_STOP_EPOCHS = 5
NUM_WORKERS = 8
PIN_MEMORY = True if torch.cuda.is_available() else False
LOAD_MODEL = True
VAL_ONLY = True
DOWNLOAD_DATA = True

def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])