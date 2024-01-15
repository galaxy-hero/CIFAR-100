# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, random_split
# import utils as u

# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#     transforms.ToTensor(),
# ])

# test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=u.DOWNLOAD_DATA, transform=transform)
# test_loader = DataLoader(test_dataset, batch_size=u.BATCH_SIZE_VAL, shuffle=False, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)

# dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=u.DOWNLOAD_DATA, transform=transform)

# train_size = len(dataset) - 10000
# val_size = 10000

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=u.BATCH_SIZE_TRAIN, shuffle=True, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)
# val_loader = DataLoader(val_dataset, batch_size=u.BATCH_SIZE_VAL, shuffle=False, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)
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