# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch
import torch.nn as nn
import torch.nn.functional as F

class Simple1DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple1DCNN, self).__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Convolutional block 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Flattening the output for the fully connected layer
        self.flattener = nn.Flatten()
        
        # After pooling three times with kernel_size=2 and stride=2, the length is reduced by a factor of 8
        # Hence, the output feature size of the last pooling layer will be 64 * (input_length / 8)
        flattened_size = 64 * (5000 // (2**3))  # 5000 / 8 = 625
        
        # Linear head
        self.fc1 = nn.Linear(flattened_size, 512)  # Adjusted according to the new flattened size
        self.fc2 = nn.Linear(512, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flattener(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
