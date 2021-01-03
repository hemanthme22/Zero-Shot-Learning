# # Testing classifiers for each values of K using saved predictions
import os
import numpy as np 
import pickle
import argparse
import pandas as pd

def eval_acc(df,colName):
    pred_check = []
    for index,row in df.iterrows():
        if row['class_name'] == row[colName]:
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
parser.add_argument("-a", "--aux_set", required=True,help=("Provide the auxilary information to use"))

args = vars(parser.parse_args())

#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
#data_set = "/"+ args['data_set']
data_set = "/"+ 'AWA2'
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"
#clustering_technique = args['clustering_technique']
clustering_technique = "gmm"

#aux_set = args['aux_set']
aux_set = 'all'

y_test = pd.read_pickle(data_loc+'/y_test.pkl')

#Running Final Predictions
y_test_df = pd.DataFrame(y_test,columns = ['class_name'])
y_test_df['rownum'] = np.arange(len(y_test_df))

#Reading CloseWord_dict pickle
closeWord_dict = pickle.load(open(data_loc+'/closeWord_dict_'+aux_set+'.pkl',"rb"))
#closeWord_dict = pickle.load(open(data_loc+'/closeWord_dict_'+'att'+'.pkl',"rb"))

#Reading Saved Classname and Attribure dictionary
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
all_classes = list(att_dict.keys())

#Running final predictions for single predictions from classifier
h = open(out_dir+data_set+"/final_accuracy_"+aux_set+"_"+clustering_technique+".txt", "w")
f = open(out_dir+data_set+"/clusterCenters_"+aux_set+"_"+ clustering_technique +".txt",'r')
lines = f.readlines()
for line in lines:
line = lines[4]

line = line.split()
modelName = line[0]
seen_classes = line[1:]
print(modelName)

unseen_classes = list(set(all_classes) - set(seen_classes))
 
#Reading the predictions for each model
pred_df = pd.read_pickle(out_dir+data_set+'/predictions_'+aux_set+'_'+clustering_technique+'/'+modelName+'.pkl')
pred_df['rownum'] = np.arange(len(pred_df))
results = pd.merge(y_test_df, pred_df, on = ['rownum'],how = 'left').drop(['rownum'],axis=1)
results['guesses'] = results['max_prob'].map(closeWord_dict)
results[['guess1','guess2']] = pd.DataFrame(results.guesses.tolist(), index= results.index)

results_seen =  results[results['class_name'].isin(seen_classes)].reset_index(drop=True)
results_unseen =  results[results['class_name'].isin(unseen_classes)].reset_index(drop=True)

results_seen['pred_check'] = (results_seen['class_name']==results_seen['guess1']).astype(int)
results_unseen['pred_check'] = (results_unseen['class_name']==results_unseen['guess2']).astype(int)

results_unseen['pred_check'].sum()

#Unseen class EDA
temp_sum = pd.DataFrame(results_unseen.groupby(['class_name'])['pred_check'].agg('sum'))
temp_counts = pd.DataFrame(results_unseen.groupby(['class_name'])['pred_check'].agg('count'))
unseen_df = temp_sum.merge(temp_counts, on = 'class_name')
unseen_df['pred_acc'] = unseen_df['pred_check_x']/unseen_df['pred_check_y']
unseen_df.sort_values(['pred_acc'])
unseen_df['pred_acc'].mean()
unseen_df.sort_values(['pred_acc'])

#results[results['guess2'] == 'hamster']

#Training set EDA
X_train = pd.read_pickle(data_loc+'/X_train.pkl')
y_train = pd.read_pickle(data_loc+'/y_train.pkl')

train_df = X_train
train_df['class_name'] = y_train
train_counts = pd.DataFrame(train_df.groupby(['class_name'])['class_name'].count())
train_counts['class_name'].mean()

#Seen class EDA
temp_sum = pd.DataFrame(results_seen.groupby(['class_name'])['pred_check'].agg('sum'))
temp_counts = pd.DataFrame(results_seen.groupby(['class_name'])['pred_check'].agg('count'))
seen_df = temp_sum.merge(temp_counts, on = 'class_name')
seen_df['pred_acc'] = seen_df['pred_check_x']/seen_df['pred_check_y']
seen_df.sort_values(['pred_acc'])
seen_df['pred_acc'].mean()
seen_df = seen_df.merge(train_counts,left_index = True,right_index = True).sort_values(['pred_acc'])
seen_df.loc[seen_df['pred_acc'] > 0.60 & seen_df['pred_acc'] < 0.90]
pd.set_option('display.max_rows', None)

f.close()
h.close()

print('Program Completed')