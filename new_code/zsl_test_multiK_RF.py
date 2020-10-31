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
data_set = "/"+ args['data_set']
#data_set = "/"+ 'AWA2'
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"
clustering_technique = args['clustering_technique']
#clustering_technique = "af"

aux_set = args['aux_set']

y_test = pd.read_pickle(data_loc+'/y_test.pkl')

#Running Final Predictions
y_test_df = pd.DataFrame(y_test,columns = ['class_name'])
y_test_df['rownum'] = np.arange(len(y_test_df))

#Reading CloseWord_dict pickle
closeWord_dict = pickle.load(open(data_loc+'/closeWord_dict_'+aux_set+'.pkl',"rb"))

#Reading Saved Classname and Attribure dictionary
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
all_classes = list(att_dict.keys())


##Running final predictions for top 3 predictions from classifier
# h = open(out_dir+data_set+"/Kmodels_final_accuracy_"+clustering_technique+".txt", "w")
# f = open(out_dir+data_set+"/clusterCenters_"+ clustering_technique +".txt",'r')
# lines = f.readlines()
# for line in lines:
#     line = line.split()
#     modelName = line[0]
#     print(modelName)
#     #Reading the predictions for each model
#     pred_df = pd.read_pickle(out_dir+data_set+'/predictions_'+clustering_technique+'/all_categories/'+modelName+'.pkl')
#     #Finding top 3 predictions
#     top_n_predictions = np.argsort(pred_df.values, axis = 1)[:,-3:]
#     #then find the associated code for each prediction
#     top_class = pred_df.columns[top_n_predictions]
#     top_class_df = pd.DataFrame(data=top_class,columns=['top1','top2','top3'])
#     top_class_df['rownum'] = np.arange(len(top_class_df))
#     results = pd.merge(y_test_df, top_class_df, on = ['rownum'],how = 'left').drop(['rownum'],axis=1)
#     results['guesses_1'] = results['top1'].map(closeWord_dict)
#     results['guesses_2'] = results['top2'].map(closeWord_dict)
#     results['guesses_3'] = results['top3'].map(closeWord_dict)
#     pred_check = []
#     for index,row in results.iterrows():
#         if (row['class_name'] in row['guesses_1']) or (row['class_name'] in row['guesses_2']) or (row['class_name'] in row['guesses_3']):
#             pred_check.append(1)
#         else:
#             pred_check.append(0)
#         
#     results['pred_check'] = pred_check
#     total_right = results['pred_check'].sum()
#     total_rows = len(pred_df)
#     accuracy = round(total_right/total_rows,4)
#     h.write(str(modelName) + ',' + str(accuracy) + '\n')
#     
# f.close()
# h.close()  

#Running final predictions for single predictions from classifier
h = open(out_dir+data_set+"/final_accuracy_"+aux_set+"_"+clustering_technique+".txt", "w")
f = open(out_dir+data_set+"/clusterCenters_"+aux_set+"_"+ clustering_technique +".txt",'r')
lines = f.readlines()
for line in lines:
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
    
    results_seen =  results[results['class_name'].isin(seen_classes)]
    results_unseen =  results[results['class_name'].isin(unseen_classes)]
    
    acc_seen = eval_acc(results_seen,'guess1')
    acc_unseen = eval_acc(results_unseen,'guess2')
    if acc_seen + acc_unseen == 0.: # avoid divide by zero error!
        h_score = 0.
    else:
        h_score = (2 * acc_seen * acc_unseen) / (acc_seen + acc_unseen)
           
    h.write(str(modelName) + ',' + str(acc_seen)+ ',' + str(acc_unseen)+ ',' + str(h_score) + '\n')
f.close()
h.close()

print('Program Completed')