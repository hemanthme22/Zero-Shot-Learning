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
from scipy.spatial import distance
from sklearn.decomposition import PCA
import math

#Function to return mahalanobis distance
def maha_dist(v, u,iv):
    return distance.mahalanobis(v, u,iv)

#Function to compute euclidean distance
def euclidean_dist(v, u):
    return distance.euclidean(v, u)

#Function to compute cosine similarity
def cosine_sim(v, u):
    return 1-distance.cosine(v, u)

#Function to apply PCA and reduce the size of features on a dictionary
def PCA_dict(x_dict):
    #Getting words and their vectors
    x_values = []
    x_keys = []
    for k, v in x_dict.items():
        x_values.append(v)
        x_keys.append(k)
    
    #Fitting PCA and transforming
    pca_ft = PCA(0.70)
    pca_ft.fit(x_values)
    x_values_pca = pca_ft.transform(x_values)
    
    pca_dict = {}
    for i in np.arange(0,len(x_keys)):
        pca_dict[x_keys[i]] = x_values_pca[i]
    
    return pca_dict

#Function to perform GMM clustering for all values of k and write cluster centers to file
def gmm_clustering(X,words,out_dir,data_set,aux_set):
    #Setting values of K for clusters
    kClusterValues= [i for i in np.arange(5,len(words),5)]	
    #Textfile to write cluster centers
    f = open(out_dir+data_set+"/clusterCenters_"+aux_set+"_gmm.txt",'w')	
    #Looping and finding cluster centers
    for k in kClusterValues:
        print(k)
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
        
#         #Creating dataframe of cluster distances to each category label
#         cluster_centers.sort()
#         cluster_info_dict = {}
#         for s in cluster_centers:
#             center_index = words.index(s)
#             distances_to_center = []
#             for w in words:
#                 word_index = words.index(w)
#                 distances_to_center.append(cosine_sim(X[word_index], X[center_index]))
#                 #distances_to_center.append(euclidean_dist(X[word_index], X[center_index]))
#                 norm = np.linalg.norm(distances_to_center)
#                 distances_to_center_normalized = distances_to_center/norm
#             
#             cluster_info_dict[s] = distances_to_center
#             #cluster_info_dict[s] = 1-distances_to_center_normalized
#             
#         cluster_df_full = pd.DataFrame(cluster_info_dict, index = words)
#         
#         #Saving Cluster Info for further use
#         g = open(out_dir+data_set+"/clusterInfo/model"+ str(len(cluster_centers))+"_clusterInfo_gmm.pkl","wb")
#         pickle.dump(cluster_df_full,g)
#         g.close()

    f.close()
    return
    
#Function to perform Affinity Propagation clustering and write centers to file
#AffinityPropagation selects best k.	
def af_clustering(X,words,out_dir,data_set,aux_set):
    af = AffinityPropagation(random_state=5,verbose = 1).fit(X)
    cluster_centers_indices = af.cluster_centers_indices_	
    #Textfile to write cluster centers
    f = open(out_dir+data_set+"/clusterCenters_"+aux_set+"_af.txt",'w')	
    cluster_centers = []
    for s in cluster_centers_indices:
        cluster_centers.append(words[s])	
    
    strK = "model" + str(len(cluster_centers_indices)) + "  " + " ".join(cluster_centers) +"\n"
    f.write(strK)
    f.close()
    #Creating dataframe of cluster information
    # cluster_centers_df = pd.DataFrame(cluster_centers,columns=['cluster_center'])
    # cluster_centers_df['cluster_label'] = np.arange(len(cluster_centers_df))
    # cluster_df = pd.DataFrame()
    # cluster_df['class_name'] = words
    # cluster_df['cluster_label'] = af.labels_
    # cluster_df_full = cluster_df.merge(cluster_centers_df,on = 'cluster_label',how = 'left').sort_values(by=['cluster_label'])
    # cluster_df_full['cluster_members'] = cluster_df_full.groupby('cluster_label')['cluster_label'].transform('size')
    
#     #Creating dataframe of cluster distances to each category label
#     cluster_centers.sort()
#     cluster_info_dict = {}
#     for s in cluster_centers:
#         center_index = words.index(s)
#         distances_to_center = []
#         for w in words:
#             word_index = words.index(w)
#             distances_to_center.append(cosine_sim(X[word_index], X[center_index]))
#             #distances_to_center.append(euclidean_dist(X[word_index], X[center_index]))
#             #norm = np.linalg.norm(distances_to_center)
#             #distances_to_center_normalized = distances_to_center/norm
#         
#         cluster_info_dict[s] = distances_to_center
#         #cluster_info_dict[s] = 1-distances_to_center_normalized
#         
#     cluster_df_full = pd.DataFrame(cluster_info_dict, index = words)
#     
#     #Saving Cluster Info for further use
#     f = open(out_dir+data_set+"/clusterInfo/model"+ str(len(cluster_centers_indices))+"_clusterInfo_af.pkl","wb")
#     pickle.dump(cluster_df_full,f)
#     f.close()
    return
    
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

#Reading Saved Classname and Attribure dictionary for clustering
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
if aux_set == "all":
    for i in list(att_dict.keys()):
        full_dict[i] = np.concatenate([att_dict[i], tax_pca_dict[i], ft_pca_dict[i]])
elif aux_set == "att":
    full_dict = att_dict
elif aux_set == "tax":
    full_dict = tax_pca_dict
elif aux_set == "ft":
    full_dict = ft_pca_dict
else:
    print("aux_set argument invalid")


#Generating close word dictionary for prediction phase
dict_keys = list(full_dict.keys())
closeWords_Count = 2
closeWord_dict = {}

for word in dict_keys:
    distance_dict = {}
    for fast_word in dict_keys:
        dist = distance.cosine(full_dict[word],full_dict[fast_word])
        distance_dict[fast_word] = dist
    closeWords_dict = {k: v for k, v in sorted(distance_dict.items(), key=lambda item: item[1])[:closeWords_Count]}
    closeWord_dict[word] = list(closeWords_dict.keys())
pickle.dump(closeWord_dict, open(data_loc+'/closeWord_dict_'+aux_set+'.pkl', 'wb'))


#Getting words and their vectors
X = []
words = []
for k, v in full_dict.items():
    X.append(v)
    words.append(k)

if clustering_technique == "gmm":
    gmm_clustering(X,words,out_dir,data_set,aux_set)
elif clustering_technique == "af":
    af_clustering(X,words,out_dir,data_set,aux_set)
else:
    print("clustering technique error")

print("Program Completed")
