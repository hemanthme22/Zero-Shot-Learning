########################################################################################################################################################
#Attribute clustering
import argparse
import pickle
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from numpy.linalg import inv
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation

#Function to return mahalanobis distance
def maha_dist(v, u,iv):
    return distance.mahalanobis(v, u,iv)

#Function to perform GMM clustering for all values of k and write cluster centers to file
def gmm_clustering(X,words,out_dir,data_set):
    #Setting values of K for clusters
    kClusterValues= [i for i in np.arange(5,len(words),5)]	
    #Textfile to write cluster centers
    f = open(out_dir+data_set+"/clusterCenters_gmm.txt",'w')	
    #Looping and finding cluster centers
    for k in kClusterValues:
        gmm = GaussianMixture(n_components=k).fit(X)
        centroids = gmm.means_
        covar_matrix = gmm.covariances_
        cluster_centers = []
        for c in np.arange(0,k):
            maha_distances = []
            for w in np.arange(0,len(X)):
                maha_distances.append(maha_dist(X[w],centroids[c],inv(covar_matrix[c])))
            minpos = maha_distances.index(min(maha_distances))
            cluster_centers.append(words[minpos])
        strK = "model" + str(k) + "  " + " ".join(cluster_centers) +"\n"
        f.write(strK)	

    f.close()
    return
	
#Function to perform Affinity Propagation clustering and write centers to file
#AffinityPropagation selects best k.	
def af_clustering(X,words,out_dir,data_set):
    af = AffinityPropagation(random_state=5,verbose = 1).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_	
    #Textfile to write cluster centers
    f = open(out_dir+data_set+"/clusterCenters_af.txt",'w')	
    cluster_centers = []
    for s in cluster_centers_indices:
        cluster_centers.append(words[s])	
    
    strK = "model" + str(len(cluster_centers_indices)) + "  " + " ".join(cluster_centers) +"\n"
    f.write(strK)
    f.close()
    #Creating dataframe of cluster information
    cluster_centers_df = pd.DataFrame(cluster_centers,columns=['cluster_center'])
    cluster_centers_df['cluster_label'] = np.arange(len(cluster_centers_df))
    cluster_df = pd.DataFrame()
    cluster_df['class_name'] = words
    cluster_df['cluster_label'] = af.labels_
    cluster_df_full = cluster_df.merge(cluster_centers_df,on = 'cluster_label',how = 'left').sort_values(by=['cluster_label'])
    cluster_df_full['cluster_members'] = cluster_df_full.groupby('cluster_label')['cluster_label'].transform('size')
    #Saving Cluster Info for further use
    f = open(out_dir+data_set+"/model"+ str(len(cluster_centers_indices))+"_clusterInfo_af.pkl","wb")
    pickle.dump(cluster_df_full,f)
    f.close()
    return
	
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs clustering of attributes for the selected dataset using selected clustering technique")
    parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
    parser.add_argument("-c", "--clustering_technique", required=True,help=("Provide the clustering technique"))
    
    args = vars(parser.parse_args())
    
    #Setting location variables
    data_dir = "/home/hd71992/thesis/new_blob/data"
    data_set = "/"+ args['data_set']
    data_loc = data_dir+data_set
    out_dir = "/home/hd71992/thesis/new_blob/outputs"
    clustering_technique = args['clustering_technique']
	
    #Reading Saved Classname and Attribure dictionary for clustering
    att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
    #print(len(att_dict))
	
    #Getting words and their vectors
    X = []
    words = []
    for k, v in att_dict.items():
        X.append(v)
        words.append(k)
	
    if clustering_technique == "gmm":
        gmm_clustering(X,words,out_dir,data_set)
    elif clustering_technique == "af":
        af_clustering(X,words,out_dir,data_set)
    else:
        print("clustering technique error")
	
    print("Program Completed")
