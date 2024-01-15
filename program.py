from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import model as m
import utils as u
import dataset as d
from multiprocessing import freeze_support
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((64, 64)),
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
    model = m.efnb0

    model = model.to(u.DEVICE)
    summary(model, input_size=(3, 32, 32))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=u.LEARNING_RATE, momentum=0.9)

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=u.DOWNLOAD_DATA, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=u.BATCH_SIZE_VAL, shuffle=False, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)

    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=u.DOWNLOAD_DATA, transform=transform, target_transform=target_transform)

    train_size = len(dataset) - 10000
    val_size = 10000

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataset = d.CachedDataset(train_dataset, device=u.DEVICE)
    val_dataset = d.CachedDataset(val_dataset, device=u.DEVICE)

    # train_loader = DataLoader(train_dataset, batch_size=u.BATCH_SIZE_TRAIN, shuffle=True, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)
    # val_loader = DataLoader(val_dataset, batch_size=u.BATCH_SIZE_VAL, shuffle=False, num_workers=u.NUM_WORKERS, pin_memory=u.PIN_MEMORY)

    train_loader = DataLoader(train_dataset, batch_size=u.BATCH_SIZE_TRAIN, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=u.BATCH_SIZE_VAL, shuffle=False, pin_memory=False)

    # config for charts and shit
    config = m.ModelConfiguration(
        epochs=u.NUM_EPOCHS, 
        learning_rate=u.LEARNING_RATE,
        batch_size_train=u.BATCH_SIZE_TRAIN,
        batch_size_val=u.BATCH_SIZE_VAL,
        device=u.DEVICE,
        num_workers=u.NUM_WORKERS,
        pin_memory=u.PIN_MEMORY,
        optimizer=torch.optim.Adam)

    train(model, optimizer, criterion, train_loader, val_loader)

def train(model, optimizer, criterion, train_loader, val_loader):
    all_outputs = []
    all_labels = []
    
    for epoch in range(u.NUM_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            #labels = labels.int()
            # encoded_labels = torch.zeros(u.BATCH_SIZE_TRAIN, 100)
            # for index, label in enumerate(labels):
            #     encoded_labels[index][label] = 1

            # inputs = inputs.to(u.DEVICE)
            # labels = encoded_labels.to(u.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            val_loss = 0.0
            for inputs, labels in val_loader:
                # inputs = inputs.to(u.DEVICE)
                # labels = labels.to(u.DEVICE)
                outputs = model(inputs)
                outputs = outputs
                
                all_labels.append(labels)
                all_outputs.append(outputs)

                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        all_outputs = torch.cat(all_outputs).argmax(dim=1)
        #print(all_outputs[:100])
        all_labels = torch.cat(all_labels).argmax(dim=1)
        #print(all_labels[:100])

        epoch_acc = u.accuracy(all_outputs, all_labels)

        print(f'Epoch [{epoch+1}/{u.NUM_EPOCHS}], Loss: {loss.item()}, Validation Loss: {val_loss}, Accuracy: {epoch_acc}')

        # early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0
        # else:
        #     counter += 1

        # if counter >= u.EARLY_STOP_EPOCHS:
        #     print(f'Early stopping after {u.EARLY_STOP_EPOCHS} epochs of no improvement.')
        #     break

if __name__ == "__main__":
    freeze_support()
    main()
