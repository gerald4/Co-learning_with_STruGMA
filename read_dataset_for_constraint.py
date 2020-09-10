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

def preprocess(data, dataset_name, random_state = 111, if_PCA = False, use_classes=False, scale = True):

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
    scaler = StandardScaler()
    #Normalising data
    if scale:

        scaler.fit(train[:,:-1])

        X_train = scaler.transform(train[:,:-1]).astype(np.float32)
        X_val = scaler.transform(val[:,:-1]).astype(np.float32)
        X_test = scaler.transform(test[:,:-1]).astype(np.float32)

    else:
        X_train = train[:,:-1].astype(np.float32)
        X_val = val[:,:-1].astype(np.float32)
        X_test = test[:,:-1].astype(np.float32)

    #if PCA, apply PCA, and then return the two first axes
    if if_PCA:
        pca = PCA(n_components=2)
        X_train = pca.fit(X_train).transform(X_train)[:,0:2].astype(np.float32)
        X_test = pca.transform(X_test)[:,0:2].astype(np.float32)
        X_val = pca.transform(X_val)[:,0:2].astype(np.float32)


    color_map = {0: "blue", 1: "red", 2: "green", 3: "yellow", 4: "orange", 5: "purple",
            6: "brown", 7: "pink", 8: "gray", 9: "olive", 10: "cyan" }

    return X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map, list(data)


def data_to_interpret(dataset_name, number = 0):

    X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map, column_names = switch_dataset(dataset_name)(if_PCA = False, number = number)



    holdout_train = np.concatenate( (np.concatenate((X_train, y_train[:,None]), axis=1), 
                                     np.concatenate((X_val, y_val[:,None]), axis=1)), axis = 0)

    return holdout_train[:,:-1], holdout_train[:,-1], X_test, y_test, np.concatenate( (y_train_onehot, y_val_onehot), axis = 0), y_test_onehot, scaler, color_map, column_names



def generate_dataset(dataset_name, number = 0):

    X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map, _ = switch_dataset(dataset_name)(if_PCA = False, number = number)



    holdout_train = np.concatenate( (np.concatenate((X_train, y_train[:,None]), axis=1), 
                                     np.concatenate((X_val, y_val[:,None]), axis=1)), axis = 0)



    np.savetxt(f"data_global/{dataset_name}/{dataset_name}_holdout_train_{number}.csv", holdout_train, delimiter=";")

    np.savetxt(f"data_global/{dataset_name}/{dataset_name}_train_{number}.csv", np.concatenate((X_train, y_train[:,None]), axis=1), delimiter=";")

    np.savetxt(f"data_global/{dataset_name}/{dataset_name}_val_{number}.csv", np.concatenate((X_val, y_val[:,None]), axis=1), delimiter=";")

    np.savetxt(f"data_global/{dataset_name}/{dataset_name}_holdout_test_{number}.csv", np.concatenate((X_test, y_test[:,None]), axis=1), delimiter=";")



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


def read_data1(path = "data_global/data1/data1.csv", if_PCA = False, number = 0):

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
def read_adult(path = "data_global/adult/", if_PCA = False, number = 0):

    #preprocessing from https://github.com/vivek2319/Predicting-US-Census-Income/blob/master/predict.py

    columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
               'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
               'Hours/Week','Native country','Income']
    train = pd.read_csv(f'{path}adult-training.csv', names=columns)
    test = pd.read_csv(f'{path}adult-test.csv', names=columns, skiprows=1)


    ####################Clean the Data

    df = pd.concat([train, test], axis=0)
    dff=df
    k=df

    df['Income'] = df['Income'].apply(lambda x: 1 if x==' >50K' else 0)

    for col in df.columns:
        if type(df[col][0]) == str:
            print("Working on " + col)
            df[col] = df[col].apply(lambda val: val.replace(" ",""))


    ####################REMOVE UNKNOWNS
        
    df.replace(' ?', np.nan, inplace=True)###making copy for visualization


    #################### Converting to int

    df = pd.concat([df, pd.get_dummies(df['Workclass'],prefix='Workclass',prefix_sep=':')], axis=1)
    df.drop('Workclass',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Marital Status'],prefix='Marital Status',prefix_sep=':')], axis=1)
    df.drop('Marital Status',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Occupation'],prefix='Occupation',prefix_sep=':')], axis=1)
    df.drop('Occupation',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Relationship'],prefix='Relationship',prefix_sep=':')], axis=1)
    df.drop('Relationship',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Race'],prefix='Race',prefix_sep=':')], axis=1)
    df.drop('Race',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Sex'],prefix='Sex',prefix_sep=':')], axis=1)
    df.drop('Sex',axis=1,inplace=True)

    df = pd.concat([df, pd.get_dummies(df['Native country'],prefix='Native country',prefix_sep=':')], axis=1)
    df.drop('Native country',axis=1,inplace=True)

    df.drop('Education', axis=1,inplace=True)

    df = df.rename(columns={'Income': 'Class'})

    print(df.shape)


    return preprocess(data = df, dataset_name = "adult", if_PCA = if_PCA, random_state=111+number)


def read_ionosphere(path="data_global/ionosphere/Ionosphere.csv", if_PCA = False, number =0):
    dataset=pd.read_csv(path,delimiter=';')
    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    dataset = dataset.rename(columns={'target': 'Class'})
    #print(dataset.head())

    return preprocess(data = dataset, dataset_name = "ionosphere", random_state=111+number )




def read_indian_liver(path="data_global/indian_liver/indian_liver.csv", if_PCA = False, number =0):
    dataset=pd.read_csv(path,na_values='?',
            names=['age',"gender"," Total_Bilirubin ","Direct_Bilirubin","Alkphos_Alkaline_Phosphotase","Sgpt_Alamine_Aminotransferase","Sgot_Aspartate Aminotransferase","TP_Total_Protiens","ALB_Albumin","A/G_Ratio_Albumin_and_Globulin_Ratio","Class"])

    dataset=dataset.dropna()
    
    dataset['gender'] = dataset["gender"].map({"Female": 0, "Male": 1})

    print(dataset.shape)

    #print(dataset.head())

    return preprocess(data = dataset, dataset_name = "indian_liver", random_state=111+number, scale = True )


def read_student_performance(path="data_global/student_performance/student-por.csv", if_PCA = False, number =0):
    dataset=pd.read_csv(path,na_values='?', delimiter=";")

    dataset=dataset.dropna()
    
    dataset['school'] = dataset["school"].map({"GP": 0, "MS": 1})

    dataset['sex'] = dataset["sex"].map({"F": 0, "M": 1})

    dataset['address'] = dataset["address"].map({"U": 0, "R": 1})

    dataset['famsize'] = dataset["famsize"].map({"LE3": 0, "GT3": 1})

    dataset['Pstatus'] = dataset["Pstatus"].map({"T": 0, "A": 1})
    #print(dataset.head())

    dataset = pd.concat([dataset, pd.get_dummies(dataset['Mjob'],prefix='Mjob',prefix_sep=':')], axis=1)
    dataset.drop('Mjob',axis=1,inplace=True)

    dataset = pd.concat([dataset, pd.get_dummies(dataset['Fjob'],prefix='Fjob',prefix_sep=':')], axis=1)
    dataset.drop('Fjob',axis=1,inplace=True)

    dataset = pd.concat([dataset, pd.get_dummies(dataset['reason'],prefix='reason',prefix_sep=':')], axis=1)
    dataset.drop('reason',axis=1,inplace=True)

    dataset = pd.concat([dataset, pd.get_dummies(dataset['guardian'],prefix='guardian',prefix_sep=':')], axis=1)
    dataset.drop('guardian',axis=1,inplace=True)


    dataset['schoolsup'] = dataset["schoolsup"].map({"yes": 1, "no": 0})

    dataset['famsup'] = dataset["famsup"].map({"yes": 1, "no": 0})

    dataset['fatherd'] = dataset["fatherd"].map({"yes": 1, "no": 0})

    dataset['activities'] = dataset["activities"].map({"yes": 1, "no": 0})

    dataset['nursery'] = dataset["nursery"].map({"yes": 1, "no": 0})

    dataset['higher'] = dataset["higher"].map({"yes": 1, "no": 0})

    dataset['internet'] = dataset["internet"].map({"yes": 1, "no": 0})

    dataset['romantic'] = dataset["romantic"].map({"yes": 1, "no": 0})

    dataset['G3'] = dataset['G3'].apply(lambda x: 0 if x< 10 else 1)

    dataset = dataset.rename(columns={'G3': 'Class'})

    print(dataset.head())

    return preprocess(data = dataset, dataset_name = "student_performance", random_state=111+number, scale = False )




def read_wdbc(if_PCA =True, path="data_global/wdbc/wdbc.data", number =0):
    names = ["ID","Class"] + [f"Feat_{i}" for i in range(30)]
    dataset=pd.read_csv(path,na_values='?',index_col=0,
            names = names)

    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    dataset = dataset.dropna()

    data = dataset[names[2:]+["Class"]]



    return preprocess(data = data, dataset_name = "wdbc", if_PCA = if_PCA, random_state=111+number)

def read_wine(if_PCA =False, path=None, number = 0):
    data = load_wine()

    dataset = pd.DataFrame(np.concatenate((data.data,data.target[:,None]),axis=1), columns = data.feature_names + ['Class'])

    

    return preprocess(data = dataset, dataset_name = "wine", random_state=111+number)

def read_pima_indian_diabetes(path="data_global/pima_indian_diabetes/diabetes.csv", if_PCA=False, number = 0):
    dataset=pd.read_csv(path,na_values='?')
#X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
#dataset=dataset.fillna(method = "bfill")
    dataset=dataset.dropna()
    dataset = dataset.rename(columns={'Outcome': 'Class'})
    #print(dataset.head())

    return preprocess(data = dataset, dataset_name = "pima_indian_diabetes", random_state=111+number)



def read_bank_marketing(path="data_global/bank_marketing/bank-additional.csv", if_PCA = False, number = 0):

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

    return preprocess(data = data_f, dataset_name = "bank_marketing", random_state=111+number)

def read_magic_gamma(path="data_global/magic_gamma/magic04.data", if_PCA = False, number = 0):

    # From https://gist.github.com/mick001/9db3609e49e98069316267349abc37b5
    features = [ 'fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'Class' ]

    dataset=pd.read_csv(path,na_values='?',sep=",", names = features)
    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    data=dataset.dropna()


    return preprocess(data = data, dataset_name = "magic_gamma", random_state= 111+number)


def read_statlog_satellite(path="data_global/statlog_satellite/statlog_satellite.csv", if_PCA = False, number=0):

    dataset=pd.read_csv(path,na_values='?',sep=";")
    #X =np.asarray(dataset.values[:,0:dataset.shape[1]]-1,dtype=np.str)
    data=dataset.dropna()


    data = data.rename(columns={"target": 'Class'})

    return preprocess(data = data, dataset_name = "statlog_satellite", random_state=111+number)


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
        "magic_gamma": read_magic_gamma,
        "statlog_satellite": read_statlog_satellite,
        "indian_liver": read_indian_liver,
        "student_performance": read_student_performance

    }

    return switcher[name]

