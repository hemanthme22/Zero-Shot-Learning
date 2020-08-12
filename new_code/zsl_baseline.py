import os
import numpy as np 
import pickle
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs clustering of attributes for the selected dataset using selected clustering technique")
    parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
    
    args = vars(parser.parse_args())
    
    #Setting location variables
    data_dir = "/home/hd71992/thesis/new_blob/data"
    data_set = "/"+ args['data_set']
    #data_set = "/"+ 'AWA2'
    data_loc = data_dir+data_set
    out_dir = "/home/hd71992/thesis/new_blob/outputs"
    
    #Reading Train and test pickles
    X_train = pd.read_pickle(data_loc+'/X_train.pkl')
    y_train = pd.read_pickle(data_loc+'/y_train.pkl')
    
    X_test = pd.read_pickle(data_loc+'/X_test.pkl')
    y_test = pd.read_pickle(data_loc+'/y_test.pkl')
    
    #Training models
    mdl_rf = RandomForestClassifier(n_estimators=1000,random_state=0,verbose=1,n_jobs=-1, min_samples_split= 2, min_samples_leaf= 1, max_features= 'auto', max_depth= 60, bootstrap= False)
    clf_fit = mdl_rf.fit(X_train, y_train)
    
    #Predicting
    yhat_clf = clf_fit.predict(X_test)

    #Accuracy
    h = open(out_dir+data_set+"/baseline_scores.txt", "w")
    print(classification_report(y_test, yhat_clf))
    print(accuracy_score(y_test, yhat_clf))
    h.write("accuracy :"+ accuracy_score(y_test, yhat_clf))
    h.write("classification report :" +classification_report(y_test, yhat_clf))
    h.close()