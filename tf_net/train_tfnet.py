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
import torch.optim.lr_scheduler as scheduler   
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score


from torch.utils.tensorboard import SummaryWriter
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
        pos_set = pd.read_csv(pos_file,header=None).iloc[1:25000:2]
        neg_set = pd.read_csv(neg_file,header=None).iloc[1:25000:2]
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
    #return pos_set,neg_set
    return data

def get_train_test(data, batchsize=1,folds=5):
    #pos_set, neg_set,
    batch_size = batchsize
    train_test_set = []

    #length = len(pos_set)
    length = len(data)
    split = 1/folds

    for i in range(folds):       
        
        test_data = data.iloc[i*(int(length*(split))):(i+1)*(int(length*(split)))]
        #test_data = pos_set.iloc[i*(int(length*(split))):(i+1)*(int(length*(split)))]
        #test_data = test_data.append(neg_set.iloc[i*(int(length*(split))):(i+1)*(int(length*(split)))])
       
        train_data = data[~data.index.isin(test_data.index)]        
        #train_data = pos_set[~pos_set.index.isin(test_data.index)]
        #train_data = train_data.append(neg_set[~neg_set.index.isin(test_data.index)])

        #test_data.drop(columns=[0],inplace=True)
        #train_data.drop(columns=[0],inplace=True)

        #test_data = test_data.sample(frac=1)
        #train_data = train_data.sample(frac=1)
        
        #test_data = test_data.reset_index()
        #train_data = train_data.reset_index()

        #print[len(train_data[train_data["label"]==0]),len(test_data[test_data["label"]==0])]
        testset = SeqDataset(test_data)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)  

        trainset = SeqDataset(train_data)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        train_test_set.append((trainloader,testloader))
    return train_test_set

# def f1_score(tp,fp,fn):
#     score = tp/(tp+0.5*(fp+fn))
#     return score

def loader_f1_score(net,loader,device):
    with torch.no_grad():
        for data in loader:
            input_key,label_key = data
            inputs = data[input_key]
            labels = data[label_key]

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)

    dis_output = [x > 0.5 for x in torch.sigmoid(outputs.data).cpu().detach().numpy()]
    f1 = f1_score(labels.cpu().detach().numpy(), dis_output, average='binary',zero_division=0)
    
    return f1

def avg_f1_score(nets,train_test_set):
    #score = 0
    for i in range(len(train_test_set)):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #tp=0
        #fp=0
        #fn=0

        _,testloader = train_test_set[i]

        with torch.no_grad():
            for data in testloader:
                input_key,label_key = data
                inputs = data[input_key]
                labels = data[label_key]

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = nets[i](inputs)

                #_, predicted = torch.max(outputs.data, 1)
                #print(outputs.data)
                #predicted = torch.sigmoid(outputs.data)

                # for label, prediction in zip(labels, predicted):
                #     pred = prediction > 0.5
                #     if label == pred:    
                #         if label == True:
                #                 tp+=1     
                #         else:
                #             if label == True:
                #                 fp+=1   
                #             if label == False:
                #                 fn+=1     

        dis_output = [x > 0.5 for x in torch.sigmoid(outputs.data).cpu().detach().numpy()]
        #print(dis_output,torch.sigmoid(outputs.data).cpu().detach().numpy(),[x > 0.5 for x in prediction],labels.cpu().detach().numpy())

        f1 = f1_score(labels.cpu().detach().numpy(), dis_output, average='binary')
        #score += tp/(tp+0.5*(fp+fn))
    
    return f1/len(train_test_set)#score/len(train_test_set),

def best_net_f1(nets,train_test_set):
    max_f1 = 0
    max_index = 0
    for i in range(len(train_test_set)):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _,testloader = train_test_set[i]

        with torch.no_grad():
            for data in testloader:
                input_key,label_key = data
                inputs = data[input_key]
                labels = data[label_key]

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = nets[i](inputs)

        dis_output = [x > 0.5 for x in torch.sigmoid(outputs.data).cpu().detach().numpy()]
        f1 = f1_score(labels.cpu().detach().numpy(), dis_output, average='binary')

        if f1 > max_f1:
            max_f1 = f1
            max_index = i
    
    return nets[max_index]

def test_accuracy(tf,net,testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0
    classes = [0,1]
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_occ = {classname: 0 for classname in classes}

    tp=0
    fp=0
    fn=0

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
                    
                    if label == True:
                        tp+=1     
                else:
                    if label == True:
                        fp+=1   
                    if label == False:
                        fn+=1                                     
                
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

    eval_text.append("F-1 score: {}".format(f1_score(tp,fp,fn)))
    return eval_text


def train(tf,net,batchsize=1,epochs=10,folds=5,subsample=False,subfolder="",learning_rate=0.001, momentum=0.9):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter('logs')    

    #pos_set,neg_set = create_data(tf,subsample)
    data = create_data(tf,subsample)
    train_test_set = get_train_test(data,batchsize=batchsize,folds=folds)
     
    nets = {}
    optimizers = {}
    #schedulers = {}
    for fold in range(len(train_test_set)):
        nets[fold] = net()
        nets[fold].to(device)
        optimizers[fold] = optim.SGD(nets[fold].parameters(), lr=learning_rate, momentum=momentum)
        #schedulers[fold] = scheduler.ReduceLROnPlateau(optimizer=optimizers[fold],mode="min",patience=12)

    #net = net()
    #net.to(device)
        
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    items = sum([len(x[0]) for x in train_test_set])

    pbar = tqdm(total=(epochs*len(train_test_set)*items*batchsize))
    loss_curve_train = []
    loss_curve_test = []
    #eval_text = []
    f1_curve_train = []
    f1_curve_test = []

    for epoch in range(epochs): 
        train_loss = []
        test_loss = []
        f1_train = []
        f1_test = []
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
                #confidence = F.softmax(outputs,dim=1)   
                loss = criterion(outputs, labels)
                loss.backward()
                optimizers[fold].step()  
                
                if fold == 0:
                    train_loss.append(loss.item())

            #schedulers[fold].step(loss)           

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
                
            

            f1_train.append(loader_f1_score(nets[fold],trainloader,device))
            f1_test.append(loader_f1_score(nets[fold],testloader,device))
            #eval_text.append(test_accuracy(tf,nets,train_test_set))

            pbar.update(items*batchsize)
        
        writer.add_graph(nets[0], inputs)
        f1_curve_train.append(np.mean(f1_train))
        f1_curve_test.append(np.mean(f1_test))

        #f1 = avg_f1_score(nets,train_test_set)
        #f1_curve.append(f1)

        loss_curve_train.extend(train_loss)
        loss_curve_test.extend(test_loss)


    #for i in range(len(train_test_set)):
    #    print("\n---- Fold ",i," -----")

    #    for txt in eval_text[i]:
    #        print(txt)
    
    print('Finished Training', tf)    
    pbar.close()    
    writer.close()
    #window_size = 100
    #loss_df = pd.DataFrame(loss_list,columns=["loss"])
    #loss_list = loss_df.rolling(window_size)
    window_size = 100

    train_ma = pd.DataFrame(loss_curve_train,columns=["loss"]).rolling(window_size)
    test_ma = pd.DataFrame(loss_curve_test,columns=["loss"]).rolling(window_size)

    loss_curve_train = train_ma.mean().values[window_size - 1:]
    loss_curve_test = test_ma.mean().values[window_size - 1:]

    plt.figure(figsize=(15,6))
    #plt.plot(loss_list.mean().values[window_size - 1:])    
    plt.plot(np.arange(0,epochs,epochs/len(loss_curve_train)),loss_curve_train, label='loss-train')
    plt.plot(np.arange(0,epochs,epochs/len(loss_curve_test)),loss_curve_test, label='loss-test')
    plt.ylabel("moving avg loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(15,6))
    plt.plot(f1_curve_train, label='f1-train')
    plt.plot(f1_curve_test, label='f1-test')
    plt.ylabel("avg f1 score")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()

    plot_final_folds(nets,train_test_set,device)

    torch.save(best_net_f1(nets,train_test_set), "models/"+subfolder+tf+".pth")


def plot_final_folds(nets,train_test_set,device):
    roc_curves = []
    pr_recalls = []

    print(len(train_test_set))
    for fold in range(len(train_test_set)): 
        _,testloader = train_test_set[fold]

        with torch.no_grad():
            for data in testloader:
                input_key,label_key = data
                inputs = data[input_key]
                labels = data[label_key]

                inputs, labels = inputs.to(device), labels.to(device)
                outputs = nets[fold](inputs)

        labels = labels.cpu().detach().numpy()
        outputs = torch.sigmoid(outputs.data).cpu().detach().numpy()

        precision, recall, thresholds = precision_recall_curve(labels,outputs)
        pr_recalls.append((recall,precision))

        fpr, tpr, thresholds = roc_curve(labels,outputs)
        roc_curves.append((fpr, tpr))

    plt.figure(figsize=(15,6))    
    for i in range(len(train_test_set)):
        plt.plot(pr_recalls[i][0],pr_recalls[i][1], marker='.', label=str(i)+"-fold")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15,6))
    for i in range(len(train_test_set)):
        plt.plot(roc_curves[i][0],roc_curves[i][1], marker='.', label=str(i)+"-fold")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rante')
    plt.legend()
    plt.show()


    


def train_cpu(tf,net,batchsize=1,epochs=10,folds=5,subsample=False,subfolder=""):    
    data = create_data(tf,subsample)
    train_test_set = get_train_test(data,batchsize=batchsize,folds=folds)
     

    net = net()
        
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
         
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())
                pbar.update(1)

            eval_text.append(test_accuracy(tf,net,testloader))  
    
    pbar.close()
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