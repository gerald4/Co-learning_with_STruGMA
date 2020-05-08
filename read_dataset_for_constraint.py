import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles

from matplotlib import pyplot as plt

import numpy as np



#Preprocessing daset

def preprocess(data, dataset_name, random_state = 111, if_PCA = False, use_classes=False):

    #Encode labels
    le = LabelEncoder()
    le.fit(data['Class'].values)
    data['Class'] = le.transform(data['Class'].values)

    #Encode for onehot
    onehot = OneHotEncoder()
    onehot.fit(data['Class'].values.reshape(-1,1))

    #Split data to train and test, and then split train to train and val
    train, test = train_test_split(data, test_size = 0.2, random_state = random_state)

    train, val = train_test_split(train, test_size = 0.2, random_state = random_state)
    train = train.values
    val = val.values
    test = test.values


    #Get y_train, y_val and y_test
    y_train = train[:,-1].astype(np.float32)
    y_val = val[:,-1].astype(np.float32)
    y_test = test[:,-1].astype(np.float32)

    #Get the one_hot_encoding-like
    y_train_onehot = onehot.transform(y_train.reshape(-1,1)).toarray().astype(np.float32)
    y_val_onehot = onehot.transform(y_val.reshape(-1,1)).toarray().astype(np.float32)
    y_test_onehot = onehot.transform(y_test.reshape(-1,1)).toarray().astype(np.float32)

    #Normalising data
    scaler = StandardScaler()
    scaler.fit(train[:,:-1])

    X_train = scaler.transform(train[:,:-1]).astype(np.float32)
    X_val = scaler.transform(val[:,:-1]).astype(np.float32)
    X_test = scaler.transform(test[:,:-1]).astype(np.float32)

    #if PCA, apply PCA, and then return the two first axes
    if if_PCA:
        pca = PCA(n_components=2)
        X_train = pca.fit(X_train).transform(X_train)[:,0:2].astype(np.float32)
        X_test = pca.transform(X_test)[:,0:2].astype(np.float32)
        X_val = pca.transform(X_val)[:,0:2].astype(np.float32)


    color_map = {-1: (1, 1, 1), 0: "blue", 1: "red", 2: "green", 3: "yellow", 4: "orange", 5: "purple",
            6: "brown", 7: "pink", 8: "gray", 9: "olive", 10: "cyan" }

    return X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map


def x_train_test(X,Y,l_classes,l_dataset, use_classes=False):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, stratify=Y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)


    C=np.unique(y_train)
    if use_classes:
        C=np.unique(l_classes)
    lb=LabelBinarizer()
    #lb.fit(C)
    lb.fit(C)
    y_train_ohot=lb.transform(y_train)
    y_val_ohot=lb.transform(y_val)
    y_test_ohot=lb.transform(y_test)
    if len(C)==2:
        y_train_ohot= np.hstack((1-y_train_ohot, y_train_ohot))
        y_test_ohot= np.hstack((1-y_test_ohot, y_test_ohot))
        y_val_ohot= np.hstack((1-y_val_ohot, y_val_ohot))

    N=X_train.shape[0]
    M=X_train.shape[1]

    return N, M, X_train, X_val, X_test, y_train, y_val, y_test #, y_train_ohot, y_val_ohot, y_test_ohot, l_classes,l_dataset


def read_data1(path = "data_global/data1/data1.csv", if_PCA = False):

    data = pd.read_csv("data_global/data1/data1.csv")

    return preprocess(data = data, dataset_name = "data1")


def read_two_info_one_clus(path = None, if_PCA = False):

    X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)

    data = pd.DataFrame(np.concatenate((X1,Y1[:,None]),axis=1), columns = ['X1', 'X2', 'Class'])

    return preprocess(data = data, dataset_name = "two_info_one_clus")

def read_two_info_one_clus_3(path = None, if_PCA = False):

    X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1, n_classes=3)

    data = pd.DataFrame(np.concatenate((X1,Y1[:,None]),axis=1), columns = ['X1', 'X2', 'Class'])

    return preprocess(data = data, dataset_name = "two_info_one_clus_3")

def read_gaussian_3_quantiles(path = None, if_PCA = False):

    X1, Y1 = make_gaussian_quantiles(n_features=2, n_classes=3)

    data = pd.DataFrame(np.concatenate((X1,Y1[:,None]),axis=1), columns = ['X1', 'X2', 'Class'])

    return preprocess(data = data, dataset_name = "two_info_one_clus_3")

   
def read_wdbc(if_PCA =True, path="data_global/wdbc/wdbc.data"):
    names = ["ID","Class"] + [f"Feat_{i}" for i in range(30)]
    dataset=pd.read_csv(path,na_values='?',index_col=0,
            names = names)

    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    dataset = dataset.dropna()

    data = dataset[names[2:]+["Class"]]



    return preprocess(data = data, dataset_name = "wdbc", if_PCA = if_PCA)



def switch_dataset(name):
    switcher={
        "wdbc": read_wdbc,
        "data1": read_data1,
        "two_info_one_clus": read_two_info_one_clus,
        "two_info_one_clus_3": read_two_info_one_clus_3,
        'read_gaussian_3_quantiles': read_gaussian_3_quantiles

    }

    return switcher[name]

