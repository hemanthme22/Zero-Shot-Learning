import fasttext
import argparse
import pickle
import numpy as np
import pandas as pd
import time

parser = argparse.ArgumentParser(description="Finds fastText word embeddings for class names related to the selected dataset")
parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))

args = vars(parser.parse_args())

#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
data_set = "/"+ args['data_set']
#data_set = "/"+ 'AWA2'
data_loc = data_dir+data_set
tax_dir = "/home/hd71992/thesis/cvpr2018-hnd-master/taxonomy"

#Reading Saved Classname and Attribure dictionary to find all classes
att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
all_classes = list(att_dict.keys())

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

#Loading fasttext english model
ft = fasttext.load_model('/home/hd71992/cc.en.300.bin')

#Creating dictionary with class names and fasttext word embedding vectors
ft_dict = {}
for i in range(len(all_classes_clean)):
    ft_dict[all_classes[i]] = ft.get_word_vector(all_classes_clean[i])
  
#Saving Classname and Fasttext dictionary for further use
g = open(data_loc+"/ft_dict.pkl","wb")
pickle.dump(ft_dict,g)
g.close()

print("Program Completed")