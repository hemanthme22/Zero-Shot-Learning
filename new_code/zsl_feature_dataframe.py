from scipy.io import loadmat
import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads Raw data in .mat format and converts to dictionary of class names and their attributes, saves as pickle")
    parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
    
    args = vars(parser.parse_args())
    
    #Setting location variables
    data_dir = "/home/hd71992/thesis/new_blob/data"
    data_set = "/"+ args['data_set']
    #data_set = "/"+ 'AWA2'
    data_loc = data_dir+data_set
    #out_dir = "/home/hd71992/thesis/new_blob/outputs"
    
    #Loading data features and class labels
    mat_feat = loadmat(data_loc+'/res101.mat')
    #mat_feat.keys()
    m_features = pd.DataFrame(mat_feat['features']).T
    m_features['labels'] = mat_feat['labels']
    #Loading class names and lookup to labels
    mat_attr = loadmat(data_loc+'/att_splits.mat')
    #mat_attr.keys()
    m_classnames = pd.DataFrame(mat_attr['allclasses_names'],columns = ['class_name'])
    m_classnames = m_classnames.applymap(lambda x: x[0])
    m_classnames['labels'] = np.arange(len(m_classnames))+1
    feat_df = m_features.merge(m_classnames,on = 'labels',how = 'left').drop(['labels'],axis=1)
    
    #Splitting entire data into train and test using stratified split
    X = feat_df.drop(['class_name'],axis=1)
    y = feat_df['class_name']
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=123,stratify = y)
    
    #Saving train and testing sets
    f = open(data_loc+"/X_train.pkl","wb")
    pickle.dump(X_train,f)
    f.close()
    
    f = open(data_loc+"/X_test.pkl","wb")
    pickle.dump(X_test,f)
    f.close()
    
    f = open(data_loc+"/y_train.pkl","wb")
    pickle.dump(y_train,f)
    f.close()
    
    f = open(data_loc+"/y_test.pkl","wb")
    pickle.dump(y_test,f)
    f.close()
    
    print('Program Completed')