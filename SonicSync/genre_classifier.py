import torch.nn as nn
import torch

class GenreClassifier(nn.Module):
    def __init__(self, num_classes, lstm_units=128, dropout_rate=0.5):
        super(GenreClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=(1, 1))
        self.flatten = nn.Flatten()
        # cnn_output_size = 128 * (N_MELS // 8) * (8 // 8)
        self.lstm = nn.LSTM(41472, lstm_units, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(lstm_units*2, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        # print(f"X shape : {x.shape}")
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, height, width)
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = x.view(batch_size, 1, -1)
        x, _ = self.lstm(x)
        x = self.dropout(torch.relu(self.fc1(x[:, -1, :])))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x