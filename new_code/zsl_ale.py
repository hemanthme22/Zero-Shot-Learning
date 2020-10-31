import argparse
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import psutil

class Compatibility(nn.Module):
    """ Attribute Label Embedding (ALE) compatibility function """
    
    def __init__(self, d_in, d_out):
        super().__init__()
        self.layer  = nn.Linear(d_in, d_out, True)
    
    def forward(self, x, s):
        x = self.layer(x)
        x = F.linear(x, s)
        return x    
        
def evaluate(model, x, y, attrs):
    """ Normalized Zero-Shot Evaluation """
    classes = torch.unique(y)
    n_class = len(classes)
    t_acc   = 0.
    y_ 		= torch.argmax(model(x, attrs), dim=1)
    for _class in classes:
        idx_sample	= [i for i, _y in enumerate(y) if _y==_class]
        n_sample 	= len(idx_sample)
        y_sample_  	= y_[idx_sample]
        y_sample   	= y[idx_sample].long()
        scr_sample	= torch.sum(y_sample_ == y_sample).item()
        acc_sample	= scr_sample / n_sample
        t_acc   	+= acc_sample
    acc = t_acc / n_class
    return acc

def extract_dict(dictionary):
    attri = []
    words = []
    for k, v in dictionary.items():
        attri.append(v)
        words.append(k)
    attri = pd.DataFrame(attri)
    return attri,words
    
#Main
parser = argparse.ArgumentParser(description="Performs attribute label embedding to find mapping between attribues and class labels")
parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
parser.add_argument("-c", "--clustering_technique", required=True,help=("Provide the clustering technique"))
    
args = vars(parser.parse_args())

FN = torch.from_numpy

optim_type = 'adam'
lr = 1e-4
wd = 1e-4
lr_decay = 0.97
n_epoch = 50
batch_size = 64

#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
data_set = "/"+ args['data_set']
#data_set = "/AWA2"
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"
clustering_technique = args['clustering_technique']
#clustering_technique = "gmm"

if torch.cuda.is_available():
    device_type = 'cuda'
    device 		= torch.device(device_type)
else: # CUDA IS NOT AVAILABLE
    device_type = 'cpu'
    device 		= torch.device(device_type)
    n_cpu = psutil.cpu_count()
    n_cpu_to_use = n_cpu // 4
    torch.set_num_threads(n_cpu_to_use)
    os.environ['MKL_NUM_THREADS'] = str(n_cpu_to_use)
    os.environ['KMP_AFFINITY'] = 'compact'

#Reading Saved Classname and Attribure dictionary
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
#print(len(att_dict))

#Reading saved training and testing sets
X_train = pd.read_pickle(data_loc+'/X_train.pkl')
y_train = pd.read_pickle(data_loc+'/y_train.pkl')
    
train_df = X_train.copy()
train_df['class_name'] = y_train
    
X_test = pd.read_pickle(data_loc+'/X_test.pkl')
y_test = pd.read_pickle(data_loc+'/y_test.pkl')

test_df = X_test.copy()
test_df['class_name'] = y_test
    
#Finding train classes for each modelK
h = open(out_dir+data_set+"/ale_accuracy_"+clustering_technique+".txt", "w")
f = open(out_dir+data_set+"/clusterCenters_"+ clustering_technique +".txt",'r')
lines = f.readlines()

for line in lines:
    #line = lines[7]
    line = line.split()
    modelName = line[0]
    seen_classes = line[1:]
    print(modelName)
    
    #Subsetting dataframe for only the classes being used now for training.
    train_now_df = train_df[train_df['class_name'].isin(seen_classes)]
    X_train_val = train_now_df.drop(['class_name'],axis=1)
    y_train_val = train_now_df['class_name'].astype('category')
    
    all_classes = list(att_dict.keys())
    unseen_classes = list(set(all_classes) - set(seen_classes))
    
    test_s_df = test_df[test_df['class_name'].isin(seen_classes)]
    X_test_s = test_s_df.drop(['class_name'],axis=1)
    y_test_s = test_s_df['class_name'].astype('category')
    
    test_u_df = test_df[test_df['class_name'].isin(unseen_classes)]
    X_test_u = test_u_df.drop(['class_name'],axis=1)
    y_test_u = test_u_df['class_name'].astype('category')
    
    #Making seen and unseen class,attribute dictionaries
    att_dict_seen = {key: att_dict[key] for key in seen_classes}
    att_dict_unseen = {key: att_dict[key] for key in unseen_classes}
    
    att,classes = extract_dict(att_dict)
    seen_att,seen_classes = extract_dict(att_dict_seen)
    unseen_att,unseen_classes = extract_dict(att_dict_unseen)
    
    #Creating class indices to push to clf
    classes_df = pd.DataFrame(classes,columns=['class_name'])
    classes_df['class_label'] = np.arange(len(classes_df))
    
    seen_classes_df = pd.DataFrame(seen_classes,columns=['class_name'])
    seen_classes_df['class_label'] = np.arange(len(seen_classes_df))
    
    unseen_classes_df = pd.DataFrame(unseen_classes,columns=['class_name'])
    unseen_classes_df['class_label'] = np.arange(len(unseen_classes_df))
    
    y_train_val_df = pd.DataFrame(y_train_val,columns=['class_name'])
    y_train_val_df_merged = y_train_val_df.merge(seen_classes_df,on = 'class_name',how = 'left')
    #y_train_val_df_merged = y_train_val_df.merge(classes_df,on = 'class_name',how = 'left')
    
    y_test_s_df = pd.DataFrame(y_test_s,columns=['class_name'])
    #y_test_s_df_merged = y_test_s_df.merge(seen_classes_df,on = 'class_name',how = 'left')
    y_test_s_df_merged = y_test_s_df.merge(classes_df,on = 'class_name',how = 'left')
    
    y_test_u_df = pd.DataFrame(y_test_u,columns=['class_name'])
    #y_test_u_df_merged = y_test_u_df.merge(unseen_classes_df,on = 'class_name',how = 'left')
    y_test_u_df_merged = y_test_u_df.merge(classes_df,on = 'class_name',how = 'left')
    
    #Tensors
    x_s_train	= FN(X_train_val.values.astype(np.float32)).type('torch.FloatTensor').to(device)
    y_s_train 	= FN(y_train_val_df_merged.class_label.values).to(device)
    
    x_s_test 	= FN(X_test_s.values.astype(np.float32)).type('torch.FloatTensor').to(device)
    y_s_test 	= FN(y_test_s_df_merged.class_label.values).to(device)
    
    x_u_test 	= FN(X_test_u.values.astype(np.float32)).type('torch.FloatTensor').to(device)
    y_u_test 	= FN(y_test_u_df_merged.class_label.values).to(device)
    
    #attr = FN(att.values).to(device).float()
    attr  = FN(att.values.astype(np.float32)).type('torch.FloatTensor').to(device)
    s_attr  = FN(seen_att.values.astype(np.float32)).type('torch.FloatTensor').to(device)
    #u_attr  = FN(unseen_att.values).to(device).float()
    
    n_s_train 	= len(x_s_train)
    
    #n_class 	= len(attr)
    #n_s_class 	= len(s_attr)
    #n_u_class	= len(u_attr)
    
    seeds = [123]
    #seeds = [123, 16] # <- Train several times randomly
    n_trials = len(seeds)
    accs = np.zeros([n_trials, n_epoch, 3], 'float32')
    
    for trial, seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # init classifier
        clf = Compatibility(d_in 	= X_train.shape[1], d_out 	= att.shape[1]).to(device)
        # init loss
        ce_loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params = clf.parameters(),lr = lr, weight_decay = wd)
        data = TensorDataset(x_s_train, y_s_train)
        data_loader	= DataLoader(data, batch_size= batch_size, shuffle=True, drop_last=False)
        for epoch_idx in range(n_epoch):
            clf.train() # Classifer train mode: ON
            running_loss = 0.
            for x, y in data_loader: # (x, y) <-> (image feature, image label)
                y_ = clf(x, s_attr)		# <- forward pass
                batch_loss = ce_loss(y_, y)		# <- calculate loss
                optimizer.zero_grad() 	# <- set gradients to zero
                batch_loss.backward()	# <- calculate gradients
                optimizer.step() 		# <- update weights
                running_loss += batch_loss.item() * batch_size # <- cumulative loss
            #scheduler.step() # <- update schedular
            epoch_loss = running_loss / n_s_train # <- calculate epoch loss
            #print("Epoch %4d\tLoss : %s" % (epoch_idx + 1, epoch_loss))
            if math.isnan(epoch_loss): continue # if loss is NAN, skip!
            
            if (epoch_idx + 1) % 1 == 0:
                clf.eval() # Classifier evaluation mode: ON
                acc_g_seen = evaluate(model = clf, x = x_s_test, y  = y_s_test, attrs = attr)
                acc_g_unseen = evaluate(model = clf, x = x_u_test, y = y_u_test, attrs = attr)
                if acc_g_seen + acc_g_unseen == 0.: # avoid divide by zero error!
                    h_score = 0.
                else:
                    h_score = (2 * acc_g_seen * acc_g_unseen) / (acc_g_seen + acc_g_unseen)        
                accs[trial, epoch_idx, :] = acc_g_seen, acc_g_unseen, h_score # <- save accuracy values
                #print("Generalized Seen acc     : %f" % acc_g_seen)
                #print("Generalized Unseen acc   : %f" % acc_g_unseen)
                #print("H-Score                  : %f" % h_score)
            
            
    gzsls_mean = accs[:, :, 0].mean(axis=0)
    #gzsls_std  = accs[:, :, 0].std(axis=0)
    gzslu_mean = accs[:, :, 1].mean(axis=0)
    #gzslu_std  = accs[:, :, 1].std(axis=0)
    gzslh_mean = accs[:, :, 2].mean(axis=0)
    #gzslh_std  = accs[:, :, 2].std(axis=0)
    
    h.write(str(modelName) + ',' + str(gzsls_mean[-1])+ ',' + str(gzslu_mean[-1])+ ',' + str(gzslh_mean[-1]) + '\n')
    
    #print ('Gzsls 	:: average: {mean:} +- {std:}'.format(mean=gzsls_mean[-1], std=gzsls_std[-1]))
    #print ('Gzslu 	:: average: {mean:} +- {std:}'.format(mean=gzslu_mean[-1], std=gzslu_std[-1]))
    #print ('Gzslh 	:: average: {mean:} +- {std:}'.format(mean=gzslh_mean[-1], std=gzslh_std[-1]))
    
print("Program Completed")