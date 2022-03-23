import torch
from torch import nn
import torch.nn.functional as F


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
        x = self.drop1(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Net_dropout_new(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.conv11 = nn.Conv2d(8, 32, 5, padding=2)
        self.batch1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 32, 5, padding=2)
        self.batch3 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 16, 5, padding=2)
        self.conv22 = nn.Conv2d(16, 8, 5, padding=2)
        self.batch2 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 10)

        self.drop1 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool(x)
        # x = self.drop1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = self.pool(x)
        # x = self.drop1(x)

        x = self.conv2(x)
        x = self.conv22(x)
        x = self.batch2(x)
        x = F.relu(x)
        # x = self.drop1(x)
        # x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

