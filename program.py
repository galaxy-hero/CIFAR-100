from time import time
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import model as m
import utils as u
import dataset as d
from multiprocessing import freeze_support
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def target_transform(label):
    encoded_label = torch.zeros(100)
    encoded_label[label] = 1
    return encoded_label

def main():
    #model = m.SimpleCNN()
    #model = m.CNNWithBatchNorm()
    #model = m.CustomModel(m.efnb0, 100)
    # model = m.build_model(
    #     pretrained=True, 
    #     fine_tune=True, 
    #     num_classes=100
    # )
    # model = m.efnb0
    model = m.get_efnb0()

    model = model.to(u.DEVICE)
    summary(model, input_size=(3, 32, 32))
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=u.LEARNING_RATE, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if u.LOAD_MODEL:
        u.load_checkpoint(torch.load("checkpoint_1705332388_acc_0.8116.tar"), model)

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=u.DOWNLOAD_DATA, transform=transform, target_transform=target_transform)
    test_loader = DataLoader(test_dataset, batch_size=u.BATCH_SIZE_VAL, shuffle=False, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)

    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=u.DOWNLOAD_DATA, transform=transform, target_transform=target_transform)

    train_size = len(dataset) - 10000
    val_size = 10000

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # train_dataset = d.CachedDataset(train_dataset, device=u.DEVICE)
    # val_dataset = d.CachedDataset(val_dataset, device=u.DEVICE)
    
    train_loader = DataLoader(train_dataset, batch_size=u.BATCH_SIZE_TRAIN, shuffle=True, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=u.BATCH_SIZE_VAL, shuffle=False, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)

    config = m.ModelConfiguration(
        epochs=u.NUM_EPOCHS, 
        learning_rate=u.LEARNING_RATE,
        batch_size_train=u.BATCH_SIZE_TRAIN,
        batch_size_val=u.BATCH_SIZE_VAL,
        device=u.DEVICE,
        num_workers=u.NUM_WORKERS,
        pin_memory=u.PIN_MEMORY,
        optimizer=torch.optim.Adam)
    if not u.VAL_ONLY:
        train(model, optimizer, criterion, train_loader, val_loader)
    else:
        val_loss, val_acc = do_validation(model, criterion, test_loader, -1)

        print(f'Validation Loss: {val_loss}, Accuracy: {val_acc}')

def do_train(model, optimizer, criterion, train_loader, epoch):
    model.train()
    
    loop = tqdm(train_loader, desc=f"Training Epoch: {epoch+1}")

    for _, (inputs, labels) in enumerate(loop):
        inputs = inputs.to(u.DEVICE)
        labels = labels.to(u.DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
    return loss

def do_validation(model, criterion, val_loader, epoch):
    model.eval()
        
    all_outputs = []
    all_labels = []
    val_loss = 0.0

    with torch.no_grad():
        loop = tqdm(val_loader, desc=f"Validation Epoch: {epoch+1}")
        
        for _, (inputs, labels) in enumerate(loop):
            inputs = inputs.to(u.DEVICE)
            labels = labels.to(u.DEVICE)
            outputs = model(inputs)
            outputs = outputs
            
            all_labels.append(labels)
            all_outputs.append(outputs)

            val_loss += criterion(outputs, labels).item()
            loop.set_postfix(loss=val_loss)


    val_loss /= len(val_loader)
    all_outputs = torch.cat(all_outputs).argmax(dim=1)
    all_labels = torch.cat(all_labels).argmax(dim=1)
    accuracy = u.accuracy(all_outputs, all_labels)

    return val_loss, accuracy

def train(model, optimizer, criterion, train_loader, val_loader):    
    for epoch in range(u.NUM_EPOCHS):
        train_loss = do_train(model, optimizer, criterion, train_loader, epoch)
        val_loss, val_acc = do_validation(model, criterion, val_loader, epoch)

        print(f'Epoch [{epoch+1}/{u.NUM_EPOCHS}], Loss: {train_loss.item()}, Validation Loss: {val_loss}, Accuracy: {val_acc}')
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        u.save_checkpoint(checkpoint, filename=f"checkpoint_{int(time())}_acc_{val_acc:.4f}.tar")

if __name__ == "__main__":
    freeze_support()
    main()
