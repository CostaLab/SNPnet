import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import glob
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.metrics import precision_recall_curve,auc
from sklearn import preprocessing

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=16, kernel_size=4,stride=1,padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=64, kernel_size=2,stride=1,padding=0)
        self.fc1 = nn.Linear(592, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, 80)
        self.fc4 = nn.Linear(80, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x.float()))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        ni = 1
        oc = 8*ni
        self.conv1 = nn.Conv2d(in_channels=ni, out_channels=oc, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=3, stride=1, padding=1)
        
        
        self.classifier = nn.Linear(in_features=oc*40*4,out_features=1)


    def forward(self, x):
        res1 = x.float()
        out = F.relu(self.conv1(res1))
        out = F.relu(self.conv2(out))

        out += res1

        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

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


class Resnet3(nn.Module):
    def __init__(self):
        super().__init__()
        ni = 1
        oc = 32*ni

        self.conv1 = nn.Conv2d(in_channels=ni, out_channels=oc, kernel_size=4, stride=1, padding='same')

        self.conv40_1 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=4, stride=1, padding='same')
        self.conv40_2 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=4, stride=1, padding='same')
        self.pool40 = nn.AvgPool2d(kernel_size=4, stride=2,padding=2)
        
        
        self.deepen = nn.Conv2d(in_channels=oc, out_channels=2*oc, kernel_size=4, stride=1, padding='same')
        oc = 2*oc 

        self.conv20_1 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=2, stride=1, padding='same')
        self.conv20_2 = nn.Conv2d(in_channels=oc, out_channels=oc, kernel_size=2, stride=1, padding='same')
        self.pool20 = nn.AvgPool2d(kernel_size=2, stride=2,padding=1)

        self.fcc1 = nn.Linear(in_features=1408,out_features=280)
        self.fcc2 = nn.Linear(in_features=280,out_features=160)
        self.fcc3 = nn.Linear(in_features=160,out_features=80)
        
        self.classifier = nn.Linear(in_features=80,out_features=1)

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
        out = self.res_block_40(out)
        out = self.res_block_40(out)

        out = self.deepen(out)
        out = self.pool40(out)

        out = self.res_block_20(out)
        out = self.res_block_20(out)

        out = self.pool20(out)
        
        out = torch.flatten(out, 1)
        out = self.fcc1(out)
        out = self.fcc2(out)
        out = self.fcc3(out)
        out = self.classifier(out)

        return out


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {'seq': self.data.iloc[idx]["seq"], 'label': self.data.iloc[idx]["label"]}
        
        return sample

def seq_to_tensor(seq):    
    seq = list(seq)
    le = preprocessing.LabelEncoder()
    le.fit(['A', 'T', 'C', 'G'])
    test = le.transform(seq)
    targets = torch.as_tensor(test,dtype=int)
    targets = F.one_hot(targets, num_classes=4)
    return targets.reshape(1,40,4)

def create_data(tf,subsample=False):
    try:
        pos_file = glob.glob('/hpcwork/rwth0776/snp-selex/deltasvm/selex_seqs/'+tf+'*_4_*.flt.fa')
        pos_file = pos_file[0]
        neg_file = glob.glob('/hpcwork/rwth0776/snp-selex/deltasvm/random_seqs/'+tf+'*_4_*.flt.fa')
        neg_file = neg_file[0]
    except:
        print("failed to load data")

    # try:
    #     pos_file = glob.glob('D:/SNPnet/Data/selex_seqs/'+tf+'*_4_*.flt.fa')
    #     pos_file = pos_file[0]
    #     neg_file = glob.glob('D:/SNPnet/Data/random_seqs/'+tf+'*_4_*.flt.fa')
    #     neg_file = neg_file[0]
    # except:
    #     print("failed to load data")
    
    if subsample:
        pos_set = pd.read_csv(pos_file,header=None).iloc[1:1000:2]
        neg_set = pd.read_csv(neg_file,header=None).iloc[1:1000:2]
    else:
        pos_set = pd.read_csv(pos_file,header=None).iloc[1::2]
        neg_set = pd.read_csv(neg_file,header=None).iloc[1::2]

    pos_set = pos_set[pos_set[0].str.contains("N")==False]
    neg_set = neg_set[neg_set[0].str.contains("N")==False]

    pos_set["seq"] = pos_set.apply(lambda x: seq_to_tensor(x[0]), axis=1)
    neg_set["seq"] = neg_set.apply(lambda x: seq_to_tensor(x[0]), axis=1)
    pos_set["label"] = torch.as_tensor(1)
    neg_set["label"] = torch.as_tensor(0)

    equalizer = min(len(pos_set),len(neg_set))
    pos_set = pos_set.sample(frac=1).iloc[:equalizer]
    neg_set = neg_set.sample(frac=1).iloc[:equalizer]

    data = pos_set.append(neg_set)
    
    data.drop(columns=[0],inplace=True)
    data = data.sample(frac=1)
    data = data.reset_index()
    return data

def get_train_test(data, batchsize=1,folds=5):
    batch_size = batchsize
    train_test_set = []
    length = len(data)
    split = 1/folds

    for i in range(folds):       
        
        test_data = data.iloc[i*(int(length*(split))):(i+1)*(int(length*(split)))]       
        train_data = data[~data.index.isin(test_data.index)]    
        testset = SeqDataset(test_data)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)  

        trainset = SeqDataset(train_data)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        train_test_set.append((trainloader,testloader))
    return train_test_set


def aupr(nets,train_test_set,device):
    pr_recalls = []
    aupr = []

    for fold in range(len(train_test_set)): 
        _,testloader = train_test_set[fold]
        predictions = []
        ground_truth = []

        with torch.no_grad():
            for data in testloader:
                input_key,label_key = data
                inputs = data[input_key]
                labels = data[label_key]

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = nets[fold](inputs)

                ground_truth.extend(list(labels.cpu().detach().numpy()))
                predictions.extend(torch.sigmoid(outputs.data).cpu().detach().numpy())

        predictions = [item for sublist in predictions for item in sublist]

        precision, recall, _ = precision_recall_curve(ground_truth,predictions)
        pr_recalls.append((recall,precision))
        aupr.append(auc(recall, precision))

    return pr_recalls,aupr

def train(tf,net,architecture,batchsize=1,epochs=10,folds=5,subsample=False,learning_rate=0.001, momentum=0.9):        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = create_data(tf,subsample)
    train_test_set = get_train_test(data,batchsize=batchsize,folds=folds)
     
    nets = {}
    optimizers = {}
    scheduler = {}
    for fold in range(len(train_test_set)):
        nets[fold] = net()
        nets[fold].to(device)
        optimizers[fold] = optim.SGD(nets[fold].parameters(), lr=learning_rate, momentum=momentum)
        scheduler[fold] = optim.lr_scheduler.ReduceLROnPlateau(optimizers[fold],patience=5)

    
    criterion = nn.BCEWithLogitsLoss()
    items = sum([len(x[0]) for x in train_test_set])

    pbar = tqdm(total=(epochs*len(train_test_set)*items*batchsize))
    loss_curve_train = []
    loss_curve_test = []
    pr = []
    au_pr = []
    window_size = 2500

    for epoch in range(epochs): 
        train_loss = []
        test_loss = []
        for fold in range(len(train_test_set)):
            trainloader,testloader = train_test_set[fold]
            for i, data in enumerate(trainloader, 0):
                input_key,label_key = data
                inputs = data[input_key]
                labels = data[label_key]
                labels = labels.unsqueeze(1)
                labels = labels.float()

                inputs, labels = inputs.to(device), labels.to(device)            
                optimizers[fold].zero_grad()

                outputs = nets[fold](inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizers[fold].step()  
                
                if fold == 0:
                    train_loss.append(loss.item())

            with torch.no_grad():
                for data in testloader:
                    input_key,label_key = data
                    inputs = data[input_key]
                    labels = data[label_key]
                    labels = labels.unsqueeze(1)
                    labels = labels.float()

                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = nets[fold](inputs)
                    loss = criterion(outputs, labels)

                    if fold == 0:
                        test_loss.append(loss.item())

            scheduler[fold].step(loss)
            pbar.update(items*batchsize)

        loss_curve_train.extend(train_loss)
        loss_curve_test.extend(test_loss)

        train_ma = pd.DataFrame(loss_curve_train,columns=["loss"]).rolling(window_size)
        test_ma = pd.DataFrame(loss_curve_test,columns=["loss"]).rolling(window_size)

        loss_curve_train_ma = list(train_ma.mean().values[window_size - 1:])
        loss_curve_test_ma = list(test_ma.mean().values[window_size - 1:])

        if epoch%5 == 0:
            pr_recalls,auc_prc = aupr(nets,train_test_set,device)

            pr.append(pr_recalls)
            au_pr.append(auc_prc)

            results = {"tf":tf,
                        "architecture":architecture,
                        "epoch": epoch,
                        "loss_curve_train": loss_curve_train_ma,
                        "loss_curve_test": loss_curve_test_ma,                        
                        "pr_recalls": pr,
                        "aupr": au_pr,
                        }

    pr_recalls,auc_prc = aupr(nets,train_test_set,device)
    pr.append(pr_recalls)
    au_pr.append(auc_prc)

    results = {
                "tf":tf,
                "architecture":architecture,
                "epoch": epoch,
                "loss_curve_train": loss_curve_train_ma,
                "loss_curve_test": loss_curve_test_ma,                        
                "pr_recalls": pr,
                "aupr": au_pr,
                }

    with open("/work/mf735606/results_resnet3/"+tf+'.pickle', 'wb') as handle:
        pickle.dump(results, handle)

    pbar.close()
    print('Finished Training', tf)    
       

    
import sys
import os

tf = sys.argv[1]
#architecture = sys.argv[2]

# if architecture == "net":
#     net=Net
# if architecture == "resnet":
#     net=Resnet
# if architecture == "resnet2":
#     net=Resnet2

net=Resnet3

train(tf,net,architecture="resnet3",batchsize=8,epochs=15,folds=5,subsample=False)