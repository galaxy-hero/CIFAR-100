
from torch.utils.data import Dataset
from torch import stack

class CachedDataset(Dataset):
    def __init__(self, dataset, cache=True, device="cpu"):
        if cache:
            data = []
            labels = []
            
            for d, l in dataset:
                data.append(d)
                labels.append(l)

            self.data = stack(data).to(device)
            self.labels = stack(labels).to(device)
            # dataset = stack([x for x in dataset]).to(device)
        # self.dataset = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]