import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import load_wine

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


    color_map = {0: "blue", 1: "red", 2: "green", 3: "yellow", 4: "orange", 5: "purple",
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

def read_adult(path = "data_global/adult/adult.data", if_PCA = False):

    df = pd.read_csv(path, 1, ",")
    df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "Country", "Class"
    ]
    df["Class"] = df["Class"].map({ "<=50K": 0, ">50K":1 })
    df['Gender'] = df['Gender'].map({' Male':1,' Female':0}).astype(int)

    df['Country'] = df['Country'].replace(' ?',np.nan)
    df['Workclass'] = df['WorkClass'].replace(' ?',np.nan)
    df['Occupation'] = df['Occupation'].replace(' ?',np.nan)
    df.dropna(how='any',inplace=True)

    df.loc[df['Country'] != ' United-States', 'Country'] = 'Non-US'
    df.loc[df['Country'] == ' United-States', 'Country'] = 'US'
    df['Country'] = df['Country'].map({'US':1,'Non-US':0}).astype(int)


    df['MaritalStatus'] = df['MaritalStatus'].replace([' Divorced',' Married-spouse-absent',' Never-married',' Separated',' Widowed'],'Single')
    df['MaritalStatus'] = df['MaritalStatus'].replace([' Married-AF-spouse',' Married-civ-spouse'],'Couple')


    df['MaritalStatus'] = df['MaritalStatus'].map({'Couple':0,'Single':1})

    rel_map = {' Unmarried':0,' Wife':1,' Husband':2,' Not-in-family':3,' Own-child':4,' Other-relative':5}

    df['Relationship'] = df['Relationship'].map(rel_map)

    race_map={' White':0,' Amer-Indian-Eskimo':1,' Asian-Pac-Islander':2,' Black':3,' Other':4}


    df['Race']= df['Race'].map(race_map)
    
    def f(x):
        if x['WorkClass'] == ' Federal-gov' or x['workclass']== ' Local-gov' or x['workclass']==' State-gov': 
            return 'govt'
        elif x['WorkClass'] == ' Private':
            return 'private'
        elif x['WorkClass'] == ' Self-emp-inc' or x['workclass'] == ' Self-emp-not-inc': 
            return 'self_employed'
        else: 
            return 'without_pay'
    
    
    df['employment_type']=df.apply(f, axis=1)

    employment_map = {'govt':0,'private':1,'self_employed':2,'without_pay':3}

    df['employment_type'] = df['employment_type'].map(employment_map)

    df.drop(labels=['Workclass','Education','Occupation'],axis=1,inplace=True)
    df.loc[(df['CapitalGain'] > 0),'CapitalGain'] = 1
    df.loc[(df['CapitalGain'] == 0 ,'CapitalGain')]= 0

    return preprocess(data = df, dataset_name = "adult", if_PCA = if_PCA)



def read_ionosphere(path="data_global/ionosphere/Ionosphere.csv", if_PCA = False):
    dataset=pd.read_csv(path,delimiter=';')
    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    dataset = dataset.rename(columns={'target': 'Class'})
    #print(dataset.head())

    return preprocess(data = dataset, dataset_name = "ionosphere")


def read_wdbc(if_PCA =True, path="data_global/wdbc/wdbc.data"):
    names = ["ID","Class"] + [f"Feat_{i}" for i in range(30)]
    dataset=pd.read_csv(path,na_values='?',index_col=0,
            names = names)

    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    dataset = dataset.dropna()

    data = dataset[names[2:]+["Class"]]



    return preprocess(data = data, dataset_name = "wdbc", if_PCA = if_PCA)

def read_pima_indian_diabetes(path="data_global/pima_indian_diabetes/diabetes.csv", if_PCA=False):
    dataset=pd.read_csv(path,na_values='?')
#X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
#dataset=dataset.fillna(method = "bfill")
    dataset=dataset.dropna()
    dataset = dataset.rename(columns={'Outcome': 'Class'})
    #print(dataset.head())

    return preprocess(data = dataset, dataset_name = "pima_indian_diabetes")

    


def read_wine(if_PCA =False, path=None):
    data = load_wine()

    dataset = pd.DataFrame(np.concatenate((data.data,data.target[:,None]),axis=1), columns = data.feature_names + ['Class'])

    

    return preprocess(data = dataset, dataset_name = "wine")

def read_bank_marketing_data(path="data_global/bank_marketing/bank-additional.csv", path_test=None,step=0):
    dataset=pd.read_csv(path,na_values='?',sep=";")
    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    dataset=dataset.dropna()
    Y =dataset["y"].as_matrix()
    dataset=dataset.drop("y",axis=1)

    l=list(dataset)
    # l.remove("gender")
    dataset=dataset.drop('duration',axis=1)
    numeric_real=["age","campaign","pdays","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]
    data=dataset[numeric_real].as_matrix(columns=None)
    enc=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='quantile')
    enc.fit(data)
    X=enc.transform(data)
    dataset[numeric_real]=pd.DataFrame(X,columns=numeric_real)


    mutiva_att=list(dataset)

    for att in mutiva_att:
        dm1=pd.get_dummies(dataset[att],drop_first=True,prefix=att)
        dataset=dataset.drop(att,axis=1)
        dataset=pd.concat([dataset,dm1],axis=1)


    xt=dataset.as_matrix(columns=None)


    X_train, X_test, y_train, y_test = train_test_split(xt, Y, test_size=0.33, random_state=step, stratify=Y)
    C=np.unique(y_train)
    lb=LabelBinarizer()
    #lb.fit(C)
    lb.fit(C)
    y_train_ohot=lb.transform(y_train)
    y_train_ohot= np.hstack((1-y_train_ohot, y_train_ohot))
    y_test_ohot=lb.transform(y_test)
    y_test_ohot= np.hstack((1-y_test_ohot, y_test_ohot))

    N=X_train.shape[0]
    M=X_train.shape[1]

    return N, M,X_train, X_test, y_train, y_test, y_train_ohot, y_test_ohot, ['Yes',"No"], list(dataset)

def read_bank_marketing(path="data_global/bank_marketing/bank-additional.csv", if_PCA = False):

    # From https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5

    dataset=pd.read_csv(path,na_values='?',sep=";")
    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    data=dataset.dropna()
    var_names = data.columns.tolist()


    categs = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
    # Quantitative vars
    quantit = [i for i in var_names if i not in categs]

    # Get dummy variables for categorical vars
    job = pd.get_dummies(data['job'])
    marital = pd.get_dummies(data['marital'])
    education = pd.get_dummies(data['education'])
    default = pd.get_dummies(data['default'])
    housing = pd.get_dummies(data['housing'])
    loan = pd.get_dummies(data['loan'])
    contact = pd.get_dummies(data['contact'])
    month = pd.get_dummies(data['month'])
    day = pd.get_dummies(data['day_of_week'])
    poutcome = pd.get_dummies(data['poutcome'])

    # Map variable to predict
    dict_map = dict()
    y_map = {'yes':1,'no':0}
    dict_map['y'] = y_map
    data = data.replace(dict_map)
    label = data['y']

    df1 = data[quantit]
    df1_names = df1.keys().tolist()

    # Scale quantitative variables
    #min_max_scaler = preprocessing.MinMaxScaler()
    #x_scaled = min_max_scaler.fit_transform(df1)
    
    #df1 = pd.DataFrame(x_scaled)
    #df1.columns = df1_names

    # Get final df
    final_df = pd.concat([df1,
                          job,
                          marital,
                          education,
                          default,
                          housing,
                          loan,
                          contact,
                          month,
                          day,
                          poutcome,
                          label], axis=1)

    data_f = final_df.rename(columns={'y': 'Class'})

    return preprocess(data = data_f, dataset_name = "bank_marketing")



def switch_dataset(name):
    switcher={
        "wdbc": read_wdbc,
        "data1": read_data1,
        "two_info_one_clus": read_two_info_one_clus,
        "two_info_one_clus_3": read_two_info_one_clus_3,
        'read_gaussian_3_quantiles': read_gaussian_3_quantiles,
        "adult": read_adult,
        "wine": read_wine,
        "ionosphere": read_ionosphere,
        "pima_indian_diabetes": read_pima_indian_diabetes,
        "bank_marketing": read_bank_marketing, 

    }

    return switcher[name]

