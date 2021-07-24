# # Training classifiers for each values of K using saved features
import os
import numpy as np 
import pickle
import argparse
import pandas as pd
import gc
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import distance


#Function to compute cosine similarity
def cosine_sim(v, u):
    return 1-distance.cosine(v, u)

def eval_acc(df):
    pred_check = []
    for index,row in df.iterrows():
        if row['class_name'] in row['guesses']:
            pred_check.append(1)
        else:
            pred_check.append(0)
    total_right = sum(pred_check)
    total_rows = len(df)
    accuracy = round(total_right/total_rows,4)
    return accuracy


#if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Performs clustering of attributes for the selected dataset using selected clustering technique")
parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
parser.add_argument("-c", "--clustering_technique", required=True,help=("Provide the clustering technique"))

args = vars(parser.parse_args())

#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
#data_set = "/"+ args['data_set']
data_set = "/"+ 'AWA2'
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"
#clustering_technique = args['clustering_technique']
clustering_technique = "af"

#Reading Train and test pickles
X_train = pd.read_pickle(data_loc+'/X_train.pkl')
y_train = pd.read_pickle(data_loc+'/y_train.pkl')

train_df = X_train
train_df['class_name'] = y_train

X_test = pd.read_pickle(data_loc+'/X_test.pkl')
y_test = pd.read_pickle(data_loc+'/y_test.pkl')

#Reading Saved Classname and Attribure dictionary
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
all_classes = list(att_dict.keys())

#Training a classifier and running predictions for each value of K.
f = open(out_dir+data_set+"/clusterCenters_"+ clustering_technique +".txt",'r')
lines = f.readlines()
for line in lines:
line = lines[0]
line = line.split()
modelName = line[0]
classesNow = line[1:]
seen_classes = line[1:]
unseen_classes = list(set(all_classes) - set(seen_classes))

print(modelName)
#Subsetting dataframe for only the classes being used now.
train_now_df = train_df[train_df['class_name'].isin(classesNow)]
X_train_val = train_now_df.drop(['class_name'],axis=1)
y_train_val = train_now_df['class_name'].astype('category')
#training randomforest
mdl_rf = RandomForestClassifier(n_estimators=1000,random_state=0,verbose=1,n_jobs=-1, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto', max_depth= 60, bootstrap= False)
clf_fit = mdl_rf.fit(X_train_val, y_train_val)

#Reading clusterInfo saved at clustering phase and converting to dictionary
cluster_info = pd.read_pickle(out_dir+data_set+"/clusterInfo/"+modelName+"_clusterInfo_"+clustering_technique+".pkl")
cluster_info_dict = cluster_info.T.to_dict('list')

# evaluate the model on train data
yhat_clf_prob = clf_fit.predict_proba(X_train_val)
pred_df = pd.DataFrame(data=yhat_clf_prob, columns=clf_fit.classes_)
pred_df_dict = pred_df.T.to_dict('list')

#Finding cosine similarities
guesses_dict = {}
for k in pred_df_dict.keys():
    similarities = {}
    for w in cluster_info_dict.keys():
        similarities[w] = cosine_sim(pred_df_dict[k], cluster_info_dict[w])
        #similarities[w] = euclidean_dist(pred_df_dict[k], cluster_info_dict[w])
    key_max = max(similarities, key=similarities.get)
    #key_max = min(similarities, key=similarities.get)
    guesses_dict[k] = key_max
    
guesses_df = pd.DataFrame.from_dict(guesses_dict, orient='index',columns=['guesses'])
guesses_df['rownum'] = np.arange(len(guesses_df))


pred_df['rownum'] = np.arange(len(pred_df))

y_train_df = pd.DataFrame(y_train_val,columns = ['class_name'])
y_train_df['rownum'] = np.arange(len(y_train_df))


results_inter = pd.merge(y_train_df, pred_df, on = ['rownum'],how = 'left')
results = pd.merge(results_inter, guesses_df, on = ['rownum'],how = 'left').drop(['rownum'],axis=1)

results_seen =  results[results['class_name'].isin(seen_classes)]
results_unseen =  results[results['class_name'].isin(unseen_classes)]

acc_seen = eval_acc(results_seen)
acc_unseen = eval_acc(results_unseen)



f.close()

print('Program Completed')