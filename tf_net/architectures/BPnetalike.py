import torch
import torch.nn as nn
import torch.nn.functional as F

class BPnetalike(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16, kernel_size=4,stride=1,padding='same')
        self.conv_d1 = nn.Conv2d(in_channels=16,out_channels=16, kernel_size=2,stride=1,dilation = 1,padding='same')
        self.conv_d2 = nn.Conv2d(in_channels=16,out_channels=16, kernel_size=2,stride=1,dilation = 2,padding='same')

        self.conv2 = nn.Conv2d(in_channels=16,out_channels=16, kernel_size=4,stride=1)

        self.fc1 = nn.Linear(592, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 16)
        self.fc4 = nn.Linear(16, 2)
        self.soft = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))

        x = F.relu(self.conv_d1(x))
        x = F.relu(self.conv_d2(x))

        x = F.relu(self.conv2(x))
            
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)

        return x