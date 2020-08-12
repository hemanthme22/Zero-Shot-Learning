import argparse
import pickle
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from numpy.linalg import inv
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

#Function to return mahalanobis distance
def maha_dist(v, u,iv):
    return distance.mahalanobis(v, u,iv)
    
#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
#data_set = "/"+ args['data_set']
data_set = "/SUN"
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"
clustering_technique = "gmm"

#Reading Saved Classname and Attribure dictionary for clustering
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
#print(len(att_dict))

#Getting words and their vectors
X = []
words = []
for k, v in att_dict.items():
    X.append(v)
    words.append(k)
        
def SelBest(arr:list, X:int)->list:
	'''
	returns the set of X configurations with shorter distance
	'''
	dx=np.argsort(arr)[:X]
	return arr[dx]

kClusterValues= [i for i in np.arange(5,len(words),5)]
n_clusters=kClusterValues
sils=[]
sils_err=[]
iterations=20
for n in n_clusters:
    tmp_sil=[]
    for _ in range(iterations):
        gmm=GaussianMixture(n, n_init=2).fit(X) 
        labels=gmm.predict(X)
        sil=metrics.silhouette_score(X, labels, metric='euclidean')
        tmp_sil.append(sil)
    val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
    err=np.std(tmp_sil)
    sils.append(val)
    sils_err.append(err)
 
 
 
import matplotlib.pyplot as plt
plt.errorbar(n_clusters, sils, yerr=sils_err)
plt.title("Silhouette Scores", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")

maxpos = sils.index(max(sils))