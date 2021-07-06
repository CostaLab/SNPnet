import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        ni = 1
        oc = 8*ni
        self.conv1 = nn.Conv2d(in_channels=ni, out_channels=oc, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        
        
        self.classifier = nn.Linear(in_features=oc*40*4,out_features=2)


    def forward(self, x):
        res1 = x.float()
        out = F.relu(self.conv1(res1))
        out = F.relu(self.conv2(out))

        out += res1

        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out