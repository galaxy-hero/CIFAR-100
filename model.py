import torch
import torch.nn as nn
from pydantic import BaseModel
from typing import Optional
from torch.utils.data import Dataset
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torchvision
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


class ModelConfiguration(BaseModel):
    epochs: int
    # dropout_rate: Optional[float]
    batch_size_train: int
    batch_size_val: int
    # layer_sizes: Optional[list[int]]
    learning_rate: float
    device: str
    num_workers: int
    pin_memory: bool
    # momentum: Optional[float]
    # weight_decay: Optional[float]
    # base_lr: Optional[float]
    # max_lr: Optional[float]
    # step_size_up: Optional[int]
    optimizer: type
    # optimizer_params: Optional[dict]
    # architecture: Optional[str] = "MLP"
    # dataset: Optional[str] = "CIFAR-100"


# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=100):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding='same')
#         self.elu = nn.ELU()

#         self.conv2 = nn.Conv2d(128, 128, kernel_size=3)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout1 = nn.Dropout(0.25)

#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
#         self.conv6 = nn.Conv2d(512, 512, kernel_size=3)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.dropout2 = nn.Dropout(0.25)

#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(512 * 4 , 1024)
#         self.dropout3 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(1024, num_classes)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.elu(self.conv1(x))
#         x = self.elu(self.conv2(x))
#         x = self.maxpool1(x)
#         x = self.elu(self.conv3(x))
#         x = self.elu(self.conv4(x))
#         x = self.maxpool2(x)
#         x = self.dropout1(x)
#         x = self.elu(self.conv5(x))
#         x = self.elu(self.conv6(x))
#         x = self.maxpool3(x)
#         x = self.dropout2(x)
#         x = self.flatten(x)
#         x = self.elu(self.fc1(x))
#         x = self.dropout3(x)
#         x = self.fc2(x)
#         x = self.softmax(x)
#         return x
    
def build_model(pretrained=True, fine_tune=True, num_classes=10):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model
    
efnb0 = EfficientNet.from_pretrained('efficientnet-b0', num_classes=100)
#efnb0 = nn.Sequential(*list(efnb0.children())[:-1]) # remove fully conn layer at the end

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict

efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
test = efficientnet_b0(weights="DEFAULT")

class CustomModel(nn.Module):
    def __init__(self, base_model, n_classes):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(efnb0._fc.in_features, n_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super(CNNWithBatchNorm, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding='same')
        self.batchnorm1 = nn.BatchNorm2d(128, momentum=0.95, eps=0.005)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding='same')
        self.batchnorm2 = nn.BatchNorm2d(128, momentum=0.95, eps=0.005)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout1 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.batchnorm3 = nn.BatchNorm2d(256, momentum=0.95, eps=0.005)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        self.batchnorm4 = nn.BatchNorm2d(256, momentum=0.95, eps=0.005)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.3)


        # self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        # self.batchnorm5 = nn.BatchNorm2d(256, momentum=0.95, eps=0.005)
        # self.relu5 = nn.ReLU()
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding='same')
        # self.batchnorm6 = nn.BatchNorm2d(256, momentum=0.95, eps=0.005)
        # self.relu6 = nn.ReLU()
        # self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        # self.dropout3 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.relu7 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.relu1(self.batchnorm1(self.conv1(x)))
        x = self.relu2(self.batchnorm2(self.conv2(x)))
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.relu3(self.batchnorm3(self.conv3(x)))
        x = self.relu4(self.batchnorm4(self.conv4(x)))
        x = self.maxpool2(x)
        x = self.dropout2(x)

        # x = self.relu5(self.batchnorm5(self.conv5(x)))
        # x = self.relu6(self.batchnorm6(self.conv6(x)))
        # x = self.maxpool3(x)
        # x = self.dropout3(x)

        x = self.flatten(x)
        x = self.relu7(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc01 = nn.Linear(64*8*8, 2048)
        self.fc02 = nn.Linear(2048, num_classes)
        self.fc1 = nn.Linear(2048, 1024)
        self.dropout3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x=self.maxpool1(x)
        # x=self.dropout3(x)
        x = self.relu(self.conv2(x))
        x=self.maxpool2(x)
        # x = self.relu(self.conv3(x))
        # x=self.maxpool3(x)
        x=self.flatten(x)
        x=self.relu(self.fc01(x))
        x=self.dropout3(x)
        x=self.relu(self.fc02(x))
        return x