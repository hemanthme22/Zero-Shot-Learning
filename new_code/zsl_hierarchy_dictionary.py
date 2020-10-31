import argparse
import pickle
import numpy as np
import pandas as pd
import time

def update_taxonomy(method, T, radius=-1, start_time=time.time()):
    num_leaves = len(T['wnids_leaf'])
    num_classes = len(T['wnids'])
    num_supers = num_classes - num_leaves
    children = T['children']
    is_ancestor_mat = T['is_ancestor_mat']
    num_children = T['num_children']
    ch_slice = T['ch_slice']
    
    if method == 'ZSL':
        root = T['root']
        label_zsl = T['label_zsl']
        
        multi_probs = np.zeros([ch_slice[-1], num_classes])
        multi_probs_class = np.zeros([num_classes, num_classes])
        multi_probs_class[root] = 1.
        for k in range(num_leaves, num_classes):
            m = k - num_leaves
            num_belong = np.sum(is_ancestor_mat[children[k]], axis=0)
            b_belong = num_belong > 0 # b_belong == is_ancestor_mat[k] except b_belong[k]
            num_belong[num_belong == 0] = 1
            multi_probs[ch_slice[m]:ch_slice[m+1]] = multi_probs_class[children[k]] = \
                b_belong * is_ancestor_mat[children[k]] / num_belong[None, :] + ~b_belong / num_children[m]
        multi_probs = multi_probs.T
        multi_probs_class = multi_probs_class.T
        
        # ideal output probabilities; see Appendix D.1
        T['multi_probs'] = multi_probs
        T['multi_probs_class'] = multi_probs_class
        T['att'] = multi_probs[label_zsl, :] # for DAG
        T['attr'] = multi_probs_class[label_zsl, :] # for tree
    else:
        print('no taxonomy update; unidentifiable method: {method}'.format(method=method))
    
    print('taxonomy for {method}; {time:8.3f} s'.format(method=method, time=time.time()-start_time))


parser = argparse.ArgumentParser(description="Performs clustering of attributes for the selected dataset using selected clustering technique")
parser.add_argument("-d", "--data_set", required=True,help=("Provide the dataset name"))

args = vars(parser.parse_args())
    
#Setting location variables
data_dir = "/home/hd71992/thesis/new_blob/data"
data_set = "/"+ args['data_set']
#data_set = "/"+ 'SUN'
data_loc = data_dir+data_set
tax_dir = "/home/hd71992/thesis/cvpr2018-hnd-master/taxonomy"

#Reading Saved Classname and Attribure dictionary to find all classes
#att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
#all_classes = list(att_dict.keys())

#Cleaning class names
# all_classes_clean = []
# if data_set == "/AWA2":
#     for item in all_classes:
#         all_classes_clean.append(item.replace("+", " "))
# elif data_set == "/SUN":
#     for item in all_classes:
#         all_classes_clean.append(item.replace("_", " "))
# elif data_set == "/CUB":
#     for item in all_classes:
#     	all_classes_clean.append(item[4:].replace("_"," "))
# else:
#     print("Undefined dataset")

if data_set == "/AWA2" or data_set == "/CUB":
    print("awa2 or cub")
    data_tax = np.load(tax_dir + data_set + '/taxonomy.npy',allow_pickle = True).item()
    update_taxonomy(method = 'ZSL',T = data_tax)
    
    tax_attr = data_tax['attr']
    data_labels = data_tax['label_zsl']
    
    f = open(tax_dir+data_set+"/allclasses.txt",'r')
    all_classes = f.read().splitlines()
    
    tax_dict = {}
    for i in range(len(all_classes)):
        tax_dict[all_classes[i]] = tax_attr[i] 
        
    #Saving Classname and Taxonomy dictionary for further use
    g = open(data_loc+"/tax_dict.pkl","wb")
    pickle.dump(tax_dict,g)
    g.close()

elif data_set == "/SUN":
    print("sun")
    att_dict = pickle.load(open(data_loc+'/att_dict.pkl',"rb"))
    all_classes = list(att_dict.keys())
    
    data_tax_df = pd.read_excel("/home/hd71992/thesis/hierarchy_three_levels/three_levels.xlsx",sheet_name = 'SUN908',header = 1)
    
    data_tax_df['category'] = data_tax_df['category'].apply(lambda x: x[4:-1].replace("/","_"))
    tax_now_df = data_tax_df[data_tax_df['category'].isin(all_classes)]
    
    tax_dict = tax_now_df.set_index('category').T.to_dict('list')
    
    #Saving Classname and Taxonomy dictionary for further use
    g = open(data_loc+"/tax_dict.pkl","wb")
    pickle.dump(tax_dict,g)
    g.close()

print('Program Completed')