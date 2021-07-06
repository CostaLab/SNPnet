import torch
from torch._C import device
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import glob
import os

from tqdm import tqdm
from sklearn import preprocessing
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt

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


#TODO Validate sample correctness
def create_data(tf,subsample=False):
    os.chdir('..')
    try:
        pos_file = glob.glob('snake/selex_seqs/'+tf+'*_4_*.flt.fa')
        pos_file = pos_file[0]
        neg_file = glob.glob('snake/random_seqs/'+tf+'*_4_*.flt.fa')
        neg_file = neg_file[0]
    except:
        os.chdir('tf_net')
    
    if subsample:
        pos_set = pd.read_csv(pos_file,header=None).iloc[1:80000:2]
        neg_set = pd.read_csv(neg_file,header=None).iloc[1:80000:2]
    else:
        pos_set = pd.read_csv(pos_file,header=None).iloc[1::2]
        neg_set = pd.read_csv(neg_file,header=None).iloc[1::2]

    pos_set = pos_set[pos_set[0].str.contains("N")==False]
    neg_set = neg_set[neg_set[0].str.contains("N")==False]
    pos_set["seq"] = pos_set.apply(lambda x: seq_to_tensor(x[0]), axis=1)
    neg_set["seq"] = neg_set.apply(lambda x: seq_to_tensor(x[0]), axis=1)
    pos_set["label"] = torch.as_tensor(1)
    neg_set["label"] = torch.as_tensor(0)

    #same number of positive/negative samples
    equalizer = min(len(pos_set),len(neg_set))
    pos_set = pos_set.sample(frac=1).iloc[:equalizer]
    neg_set = neg_set.sample(frac=1).iloc[:equalizer]

    data = pos_set.append(neg_set)
    
    data.drop(columns=[0],inplace=True)
    data = data.sample(frac=1)
    data = data.reset_index()
    os.chdir('tf_net')
    return data

def get_train_test(data,batchsize=1,folds=5):
    batch_size = batchsize
    train_test_set = []
    split = 1/folds

    for i in range(folds):
        
        test_data = data.iloc[i*(int(len(data)*(split))):(i+1)*(int(len(data)*(split)))]
        testset = SeqDataset(test_data)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

        trainset = SeqDataset(data[~data.index.isin(test_data.index)])        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        train_test_set.append((trainloader,testloader))
    return train_test_set

def test_accuracy(tf,net,testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0
    classes = [0,1]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_occ = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            input_key,label_key = data
            inputs = data[input_key]
            labels = data[label_key]

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[prediction]] += 1
                total_occ[classes[label]] += 1

    eval_text = []

    eval_text.append('\nAccuracy of the network: %d %%' % (100 * correct / total))

    for classname, correct_count in correct_pred.items():
        if total_occ[classname] > 0:
            accuracy = 100 * float(correct_count) / total_occ[classname]
        else:
            accuracy = 0.0
        eval_text.append("Accuracy for class {} with {} hits of {} is: {:.1f} %".format(classname, correct_count, total_occ[classname] ,accuracy))

    return eval_text

def train(tf,net,batchsize=1,epochs=10,folds=5,subsample=False,subfolder=""):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data = create_data(tf,subsample)
    train_test_set = get_train_test(data,batchsize=batchsize,folds=folds)
     

    net = net()
    net.to(device)
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    items = sum([len(x[0]) for x in train_test_set])

    pbar = tqdm(total=(epochs*(items)))
    loss_list = []
    eval_text = []

    for epoch in range(epochs): 
        for trainloader,testloader in train_test_set:
            for i, data in enumerate(trainloader, 0):
                
                input_key,label_key = data
                inputs = data[input_key]
                labels = data[label_key]

                inputs, labels = inputs.to(device), labels.to(device)            
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                pbar.update(1)

            eval_text.append(test_accuracy(tf,net,testloader))

    for i in range(len(train_test_set)):
        print("\n---- Fold ",i," -----")

        for txt in eval_text[i]:
            print(txt)
    
    print('Finished Training', tf)
    torch.save(net.state_dict(), "models/"+subfolder+tf+".pth")
    pbar.close()

    window_size = 100
    loss_df = pd.DataFrame(loss_list,columns=["loss"])
    loss_list = loss_df.rolling(window_size)

    plt.figure(figsize=(15,6))
    plt.plot(loss_list.mean().values[window_size - 1:])

    plt.ylabel("Cross Entropy Loss - Moving average")
    plt.xlabel("Batch")

    plt.show()

    


# def train_all(batchsize=1,split=0.2):
#     for tf in tfs:
#         train(tf,batchsize,split)

def get_eval_results(data,net,batchsize=1):
    batch_size = batchsize
    df = pd.DataFrame(data["seq"].apply(lambda x: seq_to_tensor(x)),columns=["seq"])
    df["label"] = data["seq"]
    testset = SeqDataset(df)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    pred = []
    label = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            input_key,label_key = data
            inputs = data[input_key]
            labels = data[label_key]
            
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            pred.extend(predicted)
            label.extend(labels)

    result = zip(map(lambda x: x.item(),pred),label)
    tf_net_result = pd.DataFrame(result,columns=["tf_net","seq"])
    return tf_net_result