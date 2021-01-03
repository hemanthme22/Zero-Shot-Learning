########################################################################################################################################################
#Attribute clustering
import argparse
import pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import PCA
import math
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def scatter_text(x1, y1, text_column, data1, chart_title, xlabel, ylabel):
    # Create the scatter plot
    p1 = sns.scatterplot(x=x1,y=y1, data=data1, legend=False)
    # Add text besides each point
    for line in range(0,data1.shape[0]):
         p1.text(data1[x1][line]+0.03, data1[y1][line], data1[text_column][line], horizontalalignment='left', size='medium', color='black', weight='semibold')
    # Set title and axis labels
    plt.title(chart_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return p1

#Function to apply PCA and reduce the size of features on a dictionary
def PCA_dict(x_dict,cutoff):
    #Getting words and their vectors
    x_values = []
    x_keys = []
    for k, v in x_dict.items():
        x_values.append(v)
        x_keys.append(k)
    
    #Fitting PCA and transforming
    pca_ft = PCA(cutoff)
    pca_ft.fit(x_values)
    x_values_pca = pca_ft.transform(x_values)
    
    pca_dict = {}
    for i in np.arange(0,len(x_keys)):
        pca_dict[x_keys[i]] = x_values_pca[i]
    
    return pca_dict
    
#if __name__ == "__main__":
parser = argparse.ArgumentParser(description="Performs clustering of attributes for the selected dataset using selected clustering technique")
parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))
#parser.add_argument("-c", "--clustering_technique", required=True,help=("Provide the clustering technique"))
#parser.add_argument("-a", "--aux_set", required=True,help=("Provide the auxilary information to use"))

args = vars(parser.parse_args())
    
#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
data_set = "/"+ args['data_set']
#data_set = "/"+ 'CUB'
dataS = data_set[1:]
data_loc = data_dir+data_set
out_dir = "/home/hd71992/thesis/new_blob/outputs"
#clustering_technique = args['clustering_technique']
#clustering_technique = "af"

#aux_set = args['aux_set']
aux_set = 'all'

#Reading Saved Classname and Attribure dictionary for clustering
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
all_classes = list(att_dict.keys())
#print(len(att_dict))

#Setting PCA cutoff
if data_set == "/AWA2":
    cutoff = 0.70
elif data_set == "/SUN":
    cutoff = 0.40
elif data_set == "/CUB":
    cutoff = 0.60
    
#Reading Saved Classname and Hierarchy dictionary for clustering
tax_dict = pickle.load(open(data_loc+'/tax_dict.pkl',"rb"))
#print(len(tax_dict))

tax_pca_dict = PCA_dict(tax_dict,cutoff)
print("Tax dictionary PCA from " + str(len(tax_dict[list(tax_dict.keys())[1]])) + " to " + str(len(tax_pca_dict[list(tax_dict.keys())[1]])))

#Reading Saved Classname and fastText dictionary for clustering
ft_dict = pickle.load(open(data_loc+'/ft_dict.pkl',"rb"))
#print(len(ft_dict))

ft_pca_dict = PCA_dict(ft_dict,cutoff)
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

#Cleaning class names
all_classes_clean = []
if data_set == "/AWA2":
    for item in all_classes:
        all_classes_clean.append(item.replace("+", "_"))
elif data_set == "/SUN":
    for item in all_classes:
        all_classes_clean.append(item)
elif data_set == "/CUB":
    for item in all_classes:
    	all_classes_clean.append(item[4:])
else:
    print("Undefined dataset")

#Getting words and their vectors
X = []
for k, v in full_dict.items():
    X.append(v)

tsne = TSNE(n_components=2, random_state=0)
tsne_obj= tsne.fit_transform(X)
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'classes':all_classes_clean})
tsne_df.head()

#Subsetting entries
if data_set == "/AWA2":
    tsne_subset = tsne_df
elif data_set == "/SUN" or data_set == "/CUB":
    tsne_subset = tsne_df.sample(n=10, random_state=989).reset_index(drop=True)
else:
    print("Undefined dataset")


plt.figure(figsize=(20,10))
tsne_plot = scatter_text(x1='X',y1= 'Y', text_column = 'classes', data1 = tsne_subset, chart_title = 't-SNE plot for ' +dataS, xlabel = ' ',ylabel = ' ')
tsne_plot.figure.savefig(out_dir+data_set+'/tsne_plot_'+dataS+'.png')

print("Program Completed")
