from scipy.io import loadmat
import argparse
import pickle
from numpy.linalg import inv
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA
import math

def scipy_distance(v, u):
    return distance.euclidean(v, u)

#Function to apply PCA and reduce the size of features on a dictionary
def PCA_dict(x_dict):
    #Getting words and their vectors
    x_values = []
    x_keys = []
    for k, v in x_dict.items():
        x_values.append(v)
        x_keys.append(k)
    
    #Fitting PCA and transforming
    pca_ft = PCA(.95)
    pca_ft.fit(x_values)
    x_values_pca = pca_ft.transform(x_values)
    
    pca_dict = {}
    for i in np.arange(0,len(x_keys)):
        pca_dict[x_keys[i]] = x_values_pca[i]
    
    return pca_dict

parser = argparse.ArgumentParser(description="Performs clustering of attributes for the selected dataset using selected clustering technique")
parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
parser.add_argument("-a", "--aux_set", required=True,help=("Provide the auxilary information to use"))


args = vars(parser.parse_args())

#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
data_set = "/"+ args['data_set']
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"

aux_set = args['aux_set']

#Reading Saved Classname and Attribure dictionary
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
#print(len(att_dict))

#Reading Saved Classname and Hierarchy dictionary for clustering
tax_dict = pickle.load(open(data_loc+'/tax_dict.pkl',"rb"))
#print(len(tax_dict))

tax_pca_dict = PCA_dict(tax_dict)
print("Tax dictionary PCA from " + str(len(tax_dict[list(tax_dict.keys())[1]])) + " to " + str(len(tax_pca_dict[list(tax_dict.keys())[1]])))

#Reading Saved Classname and fastText dictionary for clustering
ft_dict = pickle.load(open(data_loc+'/ft_dict.pkl',"rb"))
#print(len(ft_dict))

ft_pca_dict = PCA_dict(ft_dict)
print("fastText dictionary PCA from " + str(len(ft_dict[list(ft_dict.keys())[1]])) + " to " + str(len(ft_pca_dict[list(ft_dict.keys())[1]])))

#Creating a full dictionary from all 3 sources of auxilary information
full_dict = {}
for i in list(att_dict.keys()):
    if aux_set == "all":
        full_dict[i] = np.concatenate([att_dict[i], tax_pca_dict[i], ft_pca_dict[i]])
    elif aux_set == "att":
        full_dict[i] = np.concatenate([att_dict[i]])
    elif aux_set == "tax":
        full_dict[i] = np.concatenate([tax_pca_dict[i]])
    elif aux_set == "ft":
        full_dict[i] = np.concatenate([ft_pca_dict[i]])
    else:
        print("aux_set argument invalid")

#Generating close word dictionary for prediction phase
dict_keys = list(full_dict.keys())
closeWords_Count = 2
closeWord_dict = {}

for word in dict_keys:
    distance_dict = {}
    for fast_word in dict_keys:
        dist = scipy_distance(full_dict[word],full_dict[fast_word])
        distance_dict[fast_word] = dist
    closeWords_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])[:closeWords_Count]}
    closeWord_dict[word] = list(closeWords_dict.keys())
pickle.dump(closeWord_dict, open(data_loc+'/closeWord_dict_'+aux_set+'.pkl', 'wb'))

print('Program Completed')