# # Training classifiers for each values of K using saved features
import os
import numpy as np 
import pickle
import argparse
import pandas as pd
import gc
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

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

#Reading Train and test pickles
X_train = pd.read_pickle(data_loc+'/X_train.pkl')
y_train = pd.read_pickle(data_loc+'/y_train.pkl')

train_df = X_train
train_df['class_name'] = y_train

X_test = pd.read_pickle(data_loc+'/X_test.pkl')
y_test = pd.read_pickle(data_loc+'/y_test.pkl')

#Training a classifier and running predictions for each value of K.
f = open(out_dir+data_set+"/clusterCenters_"+aux_set+"_"+ clustering_technique +".txt",'r')
lines = f.readlines()
for line in lines:
    line = line.split()
    modelName = line[0]
    classesNow = line[1:]
    print(modelName)
    #Subsetting dataframe for only the classes being used now.
    train_now_df = train_df[train_df['class_name'].isin(classesNow)]
    X_train_val = train_now_df.drop(['class_name'],axis=1)
    y_train_val = train_now_df['class_name'].astype('category')
    #training randomforest
    mdl_rf = RandomForestClassifier(n_estimators=500,random_state=0,verbose=1,n_jobs=-1, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto', max_depth= 60, bootstrap= False)
    #mdl_rf = GradientBoostingClassifier(n_estimators=1000,random_state=0,verbose=1, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto', max_depth= 60)
    clf_fit = mdl_rf.fit(X_train_val, y_train_val)
    # evaluate the model on test data
    yhat_clf = clf_fit.predict(X_test)
    pred_df = pd.DataFrame(data=yhat_clf, columns=['max_prob'])
    pred_df.to_pickle(out_dir+data_set+'/predictions_'+aux_set+'_'+clustering_technique+'/'+modelName+'.pkl')
    #Finding prob predictions for all classes
    yhat_clf_prob = clf_fit.predict_proba(X_test)
    pred_df = pd.DataFrame(data=yhat_clf_prob, columns=clf_fit.classes_)
    pred_df.to_pickle(out_dir+data_set+'/predictions_'+aux_set+'_'+clustering_technique+'/all_categories/'+modelName+'.pkl')
    gc.collect()
f.close()

print('Program Completed')