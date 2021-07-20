import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet2(nn.Module):
    def __init__(self):
        super().__init__()
        ni = 1
        oc = 16*ni

        self.conv1 = nn.Conv2d(in_channels=ni, out_channels=oc, kernel_size=1, stride=1, padding='same')

        self.conv40_1 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=4, stride=1, padding='same')
        self.conv40_2 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=2, stride=1, padding='same')
        self.pool40 = nn.AvgPool2d(kernel_size=2, stride=2,padding=0)
        
        self.conv20_1 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=2, stride=1, padding='same')
        self.conv20_2 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=1, stride=1, padding='same')
        self.pool20 = nn.AvgPool2d(kernel_size=2, stride=2,padding=0)

        self.fcc1 = nn.Linear(in_features=160,out_features=280)
        self.fcc2 = nn.Linear(in_features=280,out_features=160)
        
        self.classifier = nn.Linear(in_features=160,out_features=1)
        #self.classifier = nn.Softmax(dim=1)



    def res_block_40(self,input):
        res = input
        out = F.relu(self.conv40_1(res))
        out = F.relu(self.conv40_2(out))
        out += res
        
        return out

    def res_block_20(self,input):
        res = input
        out = F.relu(self.conv20_1(res))
        out = F.relu(self.conv20_2(out))
        out += res
        
        return out

    def forward(self, x):
        input = x.float()
        out = F.relu(self.conv1(input))

        out = self.res_block_40(out)
        out = self.pool40(out)

        out = self.res_block_20(out)
        out = self.pool20(out)
        
        out = torch.flatten(out, 1)
        out = self.fcc1(out)
        out = self.fcc2(out)
        out = self.classifier(out)

        return out