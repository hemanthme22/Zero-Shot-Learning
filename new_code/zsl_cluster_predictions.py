# # Testing classifiers for each values of K using saved predictions
import os
import numpy as np 
import pickle
import argparse
import pandas as pd
from scipy.spatial import distance

#Function to compute euclidean distance
def euclidean_dist(v, u):
    return distance.euclidean(v, u)

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
data_set = "/"+ args['data_set']
#data_set = "/"+ 'AWA2'
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"
clustering_technique = args['clustering_technique']
#clustering_technique = "gmm"

y_test = pd.read_pickle(data_loc+'/y_test.pkl')

#Running Final Predictions
y_test_df = pd.DataFrame(y_test,columns = ['class_name'])
y_test_df['rownum'] = np.arange(len(y_test_df))

#Reading CloseWord_dict pickle
closeWord_dict = pickle.load(open(data_loc+'/closeWord_dict.pkl',"rb"))

#Reading Saved Classname and Attribure dictionary
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
all_classes = list(att_dict.keys())

#Textfile to write cluster centers
# f = open(out_dir+data_set+"/clusterCenters_gmm.txt",'w')	
# #Looping and finding cluster centers
# for k in kClusterValues:
# k=20
# gmm = GaussianMixture(n_components=k,init_params = 'random').fit(X)
# centroids = gmm.means_
# covar_matrix = gmm.covariances_
# cluster_centers = []
# 
# logprob = gmm.score_samples(X)
# preds = gmm.predict(X)
# probs = gmm.predict_proba(X)
# print(probs[:10].round(2))
# 
# for c in np.arange(0,k):
#     maha_distances = []
#     for w in np.arange(0,len(X)):
#         maha_distances.append(maha_dist(X[w],centroids[c],inv(covar_matrix[c])))
#     minpos = maha_distances.index(min(maha_distances))
#     cluster_centers.append(words[minpos])
# strK = "model" + str(k) + "  " + " ".join(cluster_centers) +"\n"
# f.write(strK)	

#Running final predictions from each cluster using new prediction method
h = open(out_dir+data_set+"/Kmodels_final_accuracy_"+clustering_technique+".txt", "w")
f = open(out_dir+data_set+"/clusterCenters_"+ clustering_technique +".txt",'r')
lines = f.readlines()
for line in lines:
#line = lines[4]
line = line.split()
modelName = line[0]
print(modelName)
seen_classes = line[1:]
unseen_classes = list(set(all_classes) - set(seen_classes))

#Reading clusterInfo saved at clustering phase and converting to dictionary
cluster_info = pd.read_pickle(out_dir+data_set+"/clusterInfo/"+modelName+"_clusterInfo_"+clustering_technique+".pkl")
cluster_info_dict = cluster_info.T.to_dict('list')

#Reading the predictions for each model
pred_df = pd.read_pickle(out_dir+data_set+'/predictions_'+clustering_technique+'/all_categories/'+modelName+'.pkl')
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
results_inter = pd.merge(y_test_df, pred_df, on = ['rownum'],how = 'left')
results = pd.merge(results_inter, guesses_df, on = ['rownum'],how = 'left').drop(['rownum'],axis=1)

results_seen =  results[results['class_name'].isin(seen_classes)]
results_unseen =  results[results['class_name'].isin(unseen_classes)]

acc_seen = eval_acc(results_seen)
acc_unseen = eval_acc(results_unseen)

if acc_seen + acc_unseen == 0.: # avoid divide by zero error!
    h_score = 0.
else:
    h_score = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)
       
h.write(str(modelName) + ',' + str(acc_seen)+ ',' + str(acc_unseen)+ ',' + str(h_score) + '\n')
f.close()
h.close()

print('Program Completed')
