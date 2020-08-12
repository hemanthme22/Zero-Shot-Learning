from scipy.io import loadmat
import argparse
import pickle
from numpy.linalg import inv
import numpy as np
import pandas as pd
from scipy.spatial import distance
import math

def scipy_distance(v, u):
    return distance.euclidean(v, u)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reads Raw data in .mat format and converts to dictionary of class names and their attributes, saves as pickle")
    parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
    
    args = vars(parser.parse_args())
    
    #Setting location variables
    data_dir = "/home/hd71992/thesis/new_blob/data"
    data_set = "/"+ args['data_set']
    data_loc = data_dir+data_set
    out_dir = "/home/hd71992/thesis/new_blob/outputs"

    #Loading data attribute features and class names
    mat_attr = loadmat(data_loc+'/att_splits.mat')
    #mat_attr.keys()
    m_classnames = mat_attr['allclasses_names']
    m_classatt = pd.DataFrame(mat_attr['att']).T

    #Creating Classname and Attribure dictionary
    att_dict = {}
    for i in np.arange(0,len(m_classnames)):
        class_name = m_classnames[i][0][0]
        class_att = m_classatt.iloc[i,:].values
        att_dict[class_name] = class_att

    #Saving Classname and Attribure dictionary for further use
    f = open(data_loc+"/att_dict.pkl","wb")
    pickle.dump(att_dict,f)
    f.close()
    
    #Generating close word dictionary for prediction phase
    dict_keys = list(att_dict.keys())
    closeWords_Count = 6
    closeWord_dict = {}
    
    for word in dict_keys:
        distance_dict = {}
        for fast_word in dict_keys:
            dist = scipy_distance(att_dict[word],att_dict[fast_word])
            distance_dict[fast_word] = dist
        closeWords_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])[:closeWords_Count]}
        closeWord_dict[word] = list(closeWords_dict.keys())
    pickle.dump(closeWord_dict, open(data_loc+'/closeWord_dict.pkl', 'wb'))

    print('Program Completed')