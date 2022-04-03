import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.batch1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.batch2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net_dropout(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.batch1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.batch2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 10)

        self.drop1 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net_dropout_new(nn.Module):
    def __init__(self, dropout_rate=0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.conv11 = nn.Conv2d(8, 32, 5, padding=2)
        self.batch1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv22 = nn.Conv2d(64, 32, 5, padding=2)
        self.batch2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv33 = nn.Conv2d(16, 8, 5, padding=2)
        self.batch3 = nn.BatchNorm2d(8)

        self.fc1 = nn.Linear(8 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 10)

        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = self.conv22(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = self.conv33(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Resnet18_1(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18_pre = models.resnet18(pretrained=True)
        self.conv_layers = nn.ModuleList()
        for child in resnet18_pre.named_children():
            if child[0] == "fc":
                break
            self.conv_layers.append(child[1])

        self.fc1 = nn.Linear(resnet18_pre.fc.in_features, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):

        for layer in self.conv_layers:
            x = layer(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Resnet18_2(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18_pre = models.resnet18(pretrained=True)

        self.conv_layers_frozen = nn.ModuleList()
        for child in resnet18_pre.named_children():
            if child[0] == "layer4":
                break
            self.conv_layers_frozen.append(child[1])

        self.layer_to_train = resnet18_pre.layer4
        self.pool = resnet18_pre.avgpool
        self.fc1 = nn.Linear(resnet18_pre.fc.in_features, 10)

    def forward(self, x):

        for layer in self.conv_layers_frozen:
            x = layer(x)
        x = self.layer_to_train(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

