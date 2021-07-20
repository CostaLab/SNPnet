import torch
import torch.nn as nn
import torch.nn.functional as F

class Fourbyone(nn.Module):
    def __init__(self):
        super().__init__()
        ni = 1
        oc = 16*ni

        self.conv1 = nn.Conv2d(in_channels=ni, out_channels=oc, kernel_size=(1,2), stride=1)
        self.conv2 = nn.Conv2d(in_channels=oc, out_channels=oc*2, kernel_size=(1,2), stride=1, padding='same')
        self.conv3 = nn.Conv2d(in_channels=oc*2, out_channels=oc*4, kernel_size=(1,2), stride=1)


        self.fcc1 = nn.Linear(in_features=5120,out_features=512)
        self.fcc2 = nn.Linear(in_features=512,out_features=256)
        self.fcc3 = nn.Linear(in_features=256,out_features=128)
        self.fcc4 = nn.Linear(in_features=128,out_features=64)

        self.classifier = nn.Linear(in_features=64,out_features=1)


    def forward(self, x):
        input = x.float()
        out = F.relu(self.conv1(input))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        out = torch.flatten(out, 1)        
        
        out = self.fcc1(out)
        out = self.fcc2(out)
        out = self.fcc3(out)
        out = self.fcc4(out)
        out = self.classifier(out)

        return out