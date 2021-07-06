import pandas as pd
import os.path
from tf_net.train_tfnet import get_eval_results
import matplotlib.pyplot as plt
import torch
import numpy as np

def get_sequence(tf):
    snps = pd.read_csv("deltasvm/data/selex_allelic_oligos_"+tf+".ref.fa",header=None).iloc[0::2]
    seqs = pd.read_csv("deltasvm/data/selex_allelic_oligos_"+tf+".ref.fa",header=None).iloc[1::2]
    sequences = pd.DataFrame(columns=["snp","seq"])
    sequences["snp"] = snps[0].values
    sequences["seq"] = seqs[0].values
    sequences["snp"] = sequences["snp"].apply(lambda x: x[1:])
    return sequences

def get_results(tf):
    results = pd.DataFrame(columns=["snp","tf","allele1_bind","allele2_bind","ref_binding","alt_binding","deltaSVM_score","preferred_allele"])
    if os.path.isfile("deltasvm/out/summary_" + tf + ".pred.tsv"):
            results = results.append(pd.read_csv("deltasvm/out/summary_" + tf + ".pred.tsv",sep='\t'), sort=False)
    results = results.rename(columns={"tf": "TF"})
    seqs = get_sequence(tf)
    results = results.merge(seqs,how='inner',on=['snp'])
    return results

def get_binding_eval_data(file,df):
    data = pd.read_csv(file)
    #True pbSNP
    data["binding"] = data["oligo_pval"]<0.05

    res = df
    data = data.merge(res, how='inner', on=["TF","snp"])
    #data = data.drop(columns=["experiment","oligo","rsid","oligo_auc","ref","alt","ref_auc","alt_auc","snp"])

    #Loss, Gain, None
    #data["gain"] = data["preferred_allele"]=="Gain"
    #data["loss"] = data["preferred_allele"]=="Loss"
    #data["none"] = data["preferred_allele"]=="None"
    #Y,N
    #data["Y"] = data["seq_binding"]=="Y"
    #data["N"] = data["seq_binding"]=="N"

    #Only (P < 0.01) = True and non-pbSNPs (P > 0.5) = False
    #data = data[(data["pval"]<0.01) | (data["pval"]>0.5)]
    return data

def print_class_acc(df,col):
    bind_cor = len(df[(df["binding"]==df[col]) & (df["binding"]==True)])
    bind_total = len(df[df["binding"]==True])
    n_bind_cor = len(df[(df["binding"]==df[col]) & (df["binding"]==False)])
    n_bind_total = len(df[df["binding"]==False])
    print("Binding accuracy: ",round(100 * float(bind_cor) / bind_total,3), "%")
    print("Non-binding accuracy: ",round(100 * float(n_bind_cor) / n_bind_total,3), "%","\n")

def print_accuary_scores(df):
    accuracy_delta = 100 * float(len(df[df["binding"]==df["ref_binding"]]) / len(df))
    print("Deltasvm accuracy ",round(accuracy_delta,3),"%")
    print_class_acc(df,"ref_binding")

    accuracy_tf_net = 100 * float(len(df[df["binding"]==df["tf_net"]]) / len(df))
    print("TF_net accuracy ",round(accuracy_tf_net,3),"%")
    print_class_acc(df,"tf_net")

    labels = ['0','1']
    binding = [len(df[df["binding"]==True]),len(df[df["binding"]==False])]
    deltasvm = [len(df[(df["binding"]==True) & (df["binding"]==df["ref_binding"])]),len(df[(df["binding"]==False) & (df["binding"]==df["ref_binding"])])]
    snpnet = [len(df[(df["binding"]==True) & (df["binding"]==df["tf_net"])]),len(df[(df["binding"]==False) & (df["binding"]==df["tf_net"])])]

    x = np.arange(len(labels))
    width = 0.15

    
    fig, ax = plt.subplots(figsize=(15,8))    

    rects1 = ax.bar(x - width, binding, width, label='Groundthruth')
    rects2 = ax.bar(x, deltasvm, width, label='deltasvm')
    rects3 = ax.bar(x + width, snpnet, width, label='snpnet')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3) 

    ax.set_ylabel('#num of occurences')
    ax.set_title('Prediction comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    #fig.tight_layout()
    plt.show()

def eval_binding_acc(tf,net,batchsize=16,subfolder=""):
    print("---",tf,"---")
    dsvm_res = get_results(tf)
    new_binding = get_binding_eval_data("deltasvm/novel_batch.csv",dsvm_res)
    new_binding.drop(["oligo_auc","oligo_pval","pbs","pbs","allele1_bind","allele2_bind"],axis=1,inplace=True)

    net = net()    
    net.load_state_dict(torch.load('tf_net/models/'+subfolder+tf+'.pth'))
    tf_net_result = get_eval_results(new_binding,net,batchsize=batchsize)

    nbc = new_binding.merge(tf_net_result,how='inner',on=["seq"])    
    nbc["ref_binding"] = nbc["ref_binding"].apply(lambda x: x=="Y")
    nbc["alt_binding"] = nbc["alt_binding"].apply(lambda x: x=="Y")
    nbc["tf_net"] = nbc["tf_net"].apply(lambda x: x==1)

    print_accuary_scores(nbc)