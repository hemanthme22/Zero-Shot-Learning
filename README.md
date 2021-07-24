# EXPLORING THE LIMITS OF ZERO-SHOT LEARNING (ZSL) : HOW LOW CAN YOU GO ?
 
Standard zero-shot learning (ZSL) methods use a large number of seen categories to predict very few unseen categories while maintaining unified data splits and evaluation metrics. This has enabled the research community to advance notably towards formulating a standard benchmark ZSL algorithm. However, the most substantial impact of ZSL lies in enabling the prediction of a large number of unseen categories from very few seen categories within a specific domain. This permits the collection and annotation of training data for only a few previously seen categories, thereby significantly mitigating the training data collection and annotation process. We address the difficult problem of predicting a large number of unseen object categories from very few previously seen categories and propose a framework that enables us to examine the limits of inferring several unseen object categories from very few previously seen object categories, i.e., the limits of ZSL. We examine the functional dependence of the classification accuracy of unseen object classes on the number and types of previously seen classes and determine the minimum number and types of previously seen classes required to achieve a prespecified classification accuracy for the unseen classes on three standard ZSL data sets. An experimental comparison of the proposed framework to a prominent ZSL technique on these data sets shows that the proposed framework achieves higher classification accuracy on average while providing valuable insights into the unseen class inference process.

## Publication
https://openaccess.thecvf.com/content/CVPR2021W/MULA/html/Dandu_Exploring_the_Limits_of_Zero-Shot_Learning_-_How_Low_Can_CVPRW_2021_paper.html

## Code Documentation & Run Instructions
Folder `new_code` contains code used for recent CVPR Workshop paper. Here, all files with the prefix `zsl_` are part of the main pipeline, others are for adhoc analysis and graphs.

### Directory structure
Make the following directory structure before running the code. Also, run code in the order mentioned below.
```
project
└───data
│   │
│   └───AWA2
│   |   │   ...
│   └───CUB
│   |   │   ...
│   └───SUN
│       │   ...
│   
└───outputs
│   │
│   └───AWA2
│   |   │   ...
│   └───CUB
│   |   │   ...
│   └───SUN
│       │   ...
```

### Data Preparation
- Image Features: reads raw data in `.mat` format and converts them into a dictionary of class names and their features(images). It also splits them into train and test sets, and saves. 

  Run command: python `zsl_feature_dataframe.py -d AWA2`

- Attribute Information(auxiliary): reads raw data in `.mat` format converts it into a dictionary of class names and their respective attributes, and saves. Raw data can be downloaded [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly).

  Run command: `python zsl_attribute_dictionary.py -d AWA2`

- Fasttext Information(auxiliary): reads raw data in `.mat` format and finds the fasttext word embeddings for each class name, and saves

  Run command: `python zsl_fasttext_dictionary.py -d AWA2`

- Hierarchial Information(auxiliary): reads raw data in `.mat` format and finds the hierarchial embeddings for each class name, and saves

  Run command: `python zsl_hierarchy_dictionary.py -d AWA2`

### Clustering
- Reads all auxiliary information from previous step, performs dimensionality reduction, and then clusters using either clustering technique. Saves clusters.

  Run command: `python zsl_clustering.py -d AWA2 -c gmm -a all`
  
- Close words: Uses generated clusters and similarity measure to find closest neighbors to cluster centers. Saved for use in testing phase.

  Run command: `python zsl_close_words_dict.py -d AWA2 -a all`

### Training & Predictions
- Train: Reads clusters and trains a RF model on cluster centers alone. Makes predictions on test and saves.

  Run command: `python zsl_train_multiK_RF.py -d AWA2 -c gmm`
  
- Test: Reads predictions from training phase, then generates alternate hypothesis based on similarity measure from clustering phase. Then, calculates H-Score.

  Run command: `python zsl_test_multiK_RF.py -d AWA2 -c gmm -a all`
  
### Comparison
- Attribute Label Embedding: Performs the attribute label embedding scheme entirely on a particular dataset.

  Run command: `python zsl_ale.py -d AWA2 -c gmm`
