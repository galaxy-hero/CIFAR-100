import torch

LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 64
NUM_EPOCHS = 350
EARLY_STOP_EPOCHS = 5
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False
LOAD_MODEL = True
DOWNLOAD_DATA = True

def accuracy(output, labels):
    fp_plus_fn = torch.logical_not(output == labels).sum().item()
    all_elements = len(output)
    return (all_elements - fp_plus_fn) / all_elements

def apply_kernel(image, kernel):
    ri, ci = image.shape       # image dimensions
    rk, ck = kernel.shape      # kernel dimensions
    ro, co = ri-rk+1, ci-ck+1  # output dimensions
    output = torch.zeros([ro, co])
    for i in range(ro): 
        for j in range(co):
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)
    return output