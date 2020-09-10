#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:18:12 2020

@author: gnanfack
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:18:12 2020
@author: gnanfack
"""


import numpy as np
import os
from time import time
import datetime
import gc
import argparse
import  pandas as pd

from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorboard.plugins.hparams import api as hp

from read_dataset_for_constraint import switch_dataset

from utils import plot_boundaries_hyperrect
from sTGMA import SoftTruncatedGaussianMixtureAnalysis

from black_box import BlackBoxNN
from config import  config_params, hyper_params
plt.rcParams["figure.figsize"] = (10,10)






@tf.function
def train_step_sTGMA(data, labels, responsibilities, eta, samples, weights, t_range):

    model.eta.assign(eta)

    with tf.GradientTape() as tape:
        expected_loglikel = model(data, labels, responsibilities, weights)
        share_loss = model.share_loss(X = samples,  black_box_model = black_box )
        loss = expected_loglikel

        prior = tfd.Independent(tfd.Normal(loc = model.m_max_min, 
                    scale =[10.]*model.data_dim), 
            reinterpreted_batch_ndims=1).log_prob(model.upper) +  tfd.Independent(
        tfd.Normal(loc = - model.m_max_min, scale =[10.]*model.data_dim), reinterpreted_batch_ndims=1).log_prob(model.lower)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer_sTGMA.apply_gradients(zip(gradients, model.trainable_variables))
    #Projected gradients
    val = tf.where(tf.less(model.lower, model.upper),
                 model.lower, (0.5)*(model.upper + model.lower - model.theta))

    val1 = tf.where(tf.less(model.lower, model.upper),
                 model.upper, (0.5)*(model.upper + model.lower + model.theta))

    model.lower.assign(val)

    model.upper.assign(val1)
    #tf.print("$$$$$$$$$$$$$$$$PROJECTION$$$$$$$$$$$$$")

    model.projection(data, labels, responsibilities, weights, t_range)

    return expected_loglikel + tf.reduce_sum(prior), share_loss #responsibilities, gradients


@tf.function
def train_step_black_box(data, labels_one_hot, samples, weights = None, _lambda = 1.):
    cross_ent = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:

        share_loss = _lambda*black_box.share_loss(X = samples,  sTGMA = model , weights = weights)
        cross_entropy = cross_ent(labels_one_hot, black_box(data))

        loss = cross_entropy + share_loss + black_box.losses()
    gradients = tape.gradient(loss , black_box.trainable_variables)

    optimizer_black_box.apply_gradients(zip(gradients, black_box.trainable_variables))

    return cross_entropy, share_loss #, gradients


@tf.function
def write_metrics(writer_use, metrics, values, step):
  with writer_use.as_default():

    for i in range(len(metrics)):
        tf.summary.scalar(metrics[i], values[i], step=step)


if __name__== "__main__":
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #             logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                
    #     except RuntimeError as e:
    #     # Memory growth must be set before GPUs have been initialized
    #         print(e)

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--dataset_name', help="the name of the dataset", default="wine")
    parser.add_argument("--n_components",help="number of components", type=int, default=3)
    parser.add_argument('--_lambda', help="lambda", type=int, default=2)
    parser.add_argument("--fold",help="fold", type=int, default=0)

    args = parser.parse_args()

    hyper_params['n_components'] = args.n_components
    config_params['fold'] = args.fold
    config_params['dataset_name'] = args.dataset_name
    hyper_params['_lambda'] = args._lambda

    converge_bb_before = config_params["converge_before"]
    np.set_printoptions(precision=5)
    tfd = tfp.distributions
    np.random.seed(config_params["seed"])
    tf.random.set_seed(config_params["seed"])


    print("TensorFlow version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))


    

    n_components = hyper_params["n_components"]
    save_loss = []


    
    dataset_name = config_params["dataset_name"]
    type_eta = hyper_params["type_eta"]
    _lambda = hyper_params["_lambda"]

    fold = config_params["fold"] 

    holdout = config_params["type"]

    print(f"Dataset: {dataset_name}, lambda: {_lambda}, n_components: {n_components}")

    data_train = np.genfromtxt(f'data_global/{dataset_name}/{dataset_name}{holdout}_train_{fold}.csv',delimiter=';')

    data_test = np.genfromtxt(f'data_global/{dataset_name}/{dataset_name}{holdout}_test_{fold}.csv',delimiter=';')

    X_train = data_train[:,:-1].astype(np.float32)

    y_train = data_train[:,-1]
    onehot = OneHotEncoder()
    onehot.fit(y_train.astype(np.int32).reshape(-1,1))

    y_train_onehot = onehot.transform(y_train.astype(np.int32).reshape(-1,1)).toarray().astype(np.float32)
    

    X_test = data_test[:,:-1].astype(np.float32)

    y_test = data_test[:,-1]

    y_test_onehot = onehot.transform(y_test.astype(np.int32).reshape(-1,1)).toarray().astype(np.float32)



    # X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map = \
    #     switch_dataset(dataset_name)(if_PCA = False)

    model = SoftTruncatedGaussianMixtureAnalysis(n_components = hyper_params["n_components"], data_dim = X_train.shape[1],
                                                 n_classes = len(np.unique(y_train)), theta = hyper_params["theta"], m_max_min = hyper_params["m_max_min"])

    model.gmm_initialisation(X_train, y_train.astype(np.int32))

    black_box = BlackBoxNN(nb_units = hyper_params["nb_units"], nb_classes =len(np.unique(y_train)))
    # tf.print(black_box.trainable_variables)

    optimizer_sTGMA = tf.optimizers.Adam(lr = hyper_params["bb_lr"])

    optimizer_black_box = tf.optimizers.Adam(lr = hyper_params["stgma_lr"])

    #exp_log_lik_loss = {}
    eta = hyper_params["value_eta"]
    tol = hyper_params["tol"]
    loss2 = 100000.0
    save_loss1 = []
    save_loss2 = []
    save_share_loss1 = []
    save_share_loss2 = []

    list_train_acc_bb = []
    list_train_acc_stgma = []
    list_fidel_train = []
    list_nmi_train = []

    list_nmi_test = []
    list_fidel_test = []
    list_test_acc_bb = []
    list_test_acc_stgma = []

    metrics_train = ['acc_nn', 
                'acc_stgma', 
                'fidelity',
                'nmi',
                'cross_entropy_vs_shareloss1',
                'expected_loglikel_vs_shareloss2']

    # metrics_test = ['acc_nn_test', 'acc_stgma_test',
    # 'fidel_test', 'share_loss1',  'share_loss2'
    # ]

    diff = []
    directory = f"results/holdout/{dataset_name}/components_{n_components}_lambda_{str(_lambda)}_{str(hyper_params['nb_units'])}"

    if converge_bb_before:
        if config_params["weights"]:
            directory = f"{directory}∕converge_bbb_before_with_weights"
        else:
            directory = f"{directory}∕converge_bbb_before"
    elif config_params["weights"]:
        directory = f"{directory}_with_weights"



    os.makedirs(directory, exist_ok = True)
    directory = f"{directory}/fold_{fold}"
    os.makedirs(directory, exist_ok = True)
    
    directory = f"{directory}/"

    log_dir = f"{directory}/logs"
    os.makedirs(log_dir, exist_ok = True)



    # writer_train = tf.summary.create_file_writer(f"{log_dir}/train")
    # writer_test = tf.summary.create_file_writer(f"{log_dir}/test")
    writer_train = tf.summary.create_file_writer(f"{log_dir}/train")
    writer_test = tf.summary.create_file_writer(f"{log_dir}/test")


    # directory = f"{directory}/value_{eta}"


    # dataset = tf.data.Dataset.zip((X_train, y_train_onehot)) 
    if converge_bb_before:
        for j in range(50):
            print(f"Black-box update iteration {j}")
            loss1, share_loss1 = train_step_black_box(data = X_train,
                                                      labels_one_hot = y_train_onehot, samples = tf.constant(X_train),
                                                      weights = None,
                                                      _lambda = tf.constant(0.))
            #loss1, share_loss1 = loss1.numpy(), share_loss1.numpy()


    for step in range(hyper_params["global_steps"]):
        #Expectation step
        if type_eta == "eta_variant":
            eta = (0.5)*np.sqrt(step) + eta



        filename = f"{directory}boundaries_importance_{step}.png"
        responsibilities = model.compute_responsibilities(X_train, y_train.astype(np.int32))


        #Maximization
        print("Sampling ...")
        # samples = model.sample_directly(nb_samples = 500).numpy()
        # weights = None
        samples = model.sample_importance(nb_samples = X_train.shape[0]*hyper_params["nb_MC_samples"])
        weights = None

        #print(weights)

        #exp_log_lik_loss[step] = []
        
        #Learning black-box model
        for j in range(hyper_params["bb_steps"]):
            print(f"Black-box update iteration {j}")
            loss1, share_loss1 = train_step_black_box(data = X_train,
                                                      labels_one_hot = y_train_onehot, samples = samples,
                                                      weights = weights,
                                                      _lambda = _lambda)
            loss1, share_loss1 = loss1.numpy(), share_loss1.numpy()

        save_loss1.append(loss1)
        save_share_loss1.append(share_loss1)

        black_box_probs = black_box(X_train)
        black_box_labels = np.argmax(black_box_probs.numpy(), axis = 1)

        if not(config_params["weights"]):
            black_box_probs = tf.one_hot(black_box_labels, len(np.unique(y_train)))


        #Learning sTGMA
        #tf.print("----Begin----")
        for j in range(hyper_params["stgma_steps"]):

            loss, share_loss2 = train_step_sTGMA(data = tf.constant(X_train), labels = tf.constant(black_box_labels.astype(np.int32)),
                                    responsibilities = responsibilities, eta = tf.Variable(eta, trainable = False, dtype=tf.float32), 
                                    samples = samples, weights = black_box_probs,  

                                    t_range = tf.random.shuffle(tf.range(tf.constant(model.data_dim)), seed=(2^step)*(2*j+1) ))

            loss, share_loss2 = loss.numpy(), share_loss2.numpy() #, resp.numpy()

        #print("---->Projection<----")
        #model.projection(X_train, responsibilities)
        #print(model.no_ovelap_test())
        #print(tf.where(tf.less(model.no_ovelap_test(),0)))
        #print(model.upper- model.lower)
            #exp_log_lik_loss[step].append(loss)
        save_loss2.append(loss)
        save_share_loss2.append(share_loss2)


        y_train_bb = np.argmax(black_box.predict(X_train).numpy(), axis = 1)
        y_test_bb = np.argmax(black_box.predict(X_test).numpy(), axis = 1)
        y_train_model = model.predict(X_train).numpy()
        y_test_model =  model.predict(X_test).numpy()

        train_acc_bb = accuracy_score(y_train, y_train_bb)
        train_acc_stgma = accuracy_score(y_train, y_train_model)

        test_acc_bb = accuracy_score(y_test, y_test_bb)
        test_acc_stgma = accuracy_score(y_test, y_test_model)

        test_fidel = accuracy_score(y_test_model, y_test_bb)
        test_nmi = normalized_mutual_info_score(y_test_model, y_test_bb)

        train_fidel = accuracy_score(y_train_model, y_train_bb)
        train_nmi = normalized_mutual_info_score(y_train_model, y_train_bb)

        list_fidel_train.append(train_fidel)
        list_nmi_train.append(train_nmi)
        list_train_acc_bb.append(train_acc_bb)
        list_train_acc_stgma.append(train_acc_stgma)


        list_fidel_test.append(test_fidel)
        list_nmi_test.append(test_nmi)
        list_test_acc_bb.append(test_acc_bb)
        list_test_acc_stgma.append(test_acc_stgma)


        values_train = [train_acc_bb, train_acc_stgma,
                train_fidel, train_nmi, loss1, 
                loss]
        values_test = [test_acc_bb,
        test_acc_stgma, test_fidel, test_nmi,
        share_loss1, share_loss2] 
        
        write_metrics(writer_test, metrics_train, tf.Variable(values_test, trainable = False), tf.constant(step, tf.int64))
        write_metrics(writer_train, metrics_train, tf.Variable(values_train, trainable = False), tf.constant(step, tf.int64))
        


        #writer_test.flush()
        if model.data_dim ==2:
            plot_boundaries_hyperrect(X = X_train,
                                   y = y_train,
                                   x_axis= 0,
                                   y_axis= 1,
                                   black_box= black_box,
                                   color_map= color_map,
                                   file_name = filename,
                                   sTGMA= model,
                                   steps= 100)



        if np.abs(loss2 - loss)/loss2 < tol and _lambda!=0:
            break
        else:
            diff.append(loss2-loss)
            loss2 = loss
        print(f"***** Iteration {step} *****")
        print(f"loss1: {loss1}, shareloss1: {share_loss1}, loss2: {loss2}, shareloss2: {share_loss2}")
    #    plot_pdfR(X_train[:,0], X_train[:,1], f"{filename}_density_{i}.png", model, color_map)

        gc.collect()
        del responsibilities
        del samples


df = pd.DataFrame({"cross_entropy": save_loss1,
                    "shareloss1": save_share_loss1,
                    "shareloss2": save_share_loss2,
                    "expected_loglikel": save_loss2,
                    "train_acc_bb": list_train_acc_bb,
                    "test_acc_bb": list_test_acc_bb,
                    "train_acc_stgma": list_train_acc_stgma,
                    "test_acc_stgma": list_test_acc_stgma,
                    "train_fidel": list_fidel_train,
                    "test_fidel": list_fidel_test,
                    "train_nmi": list_nmi_train,
                    "test_nmi": list_nmi_test})
df.to_csv(f"{directory}/{dataset_name}_{str(_lambda)}_{str(n_components)}_holdout.csv")

print('----------Training----------')
print(f"accuracy black box: {train_acc_bb}")
print(f"accuracy sTGMA: {train_acc_stgma}")
print(f"Fidelity: {train_fidel}")

print('----------Testing----------')
print(f"accuracy black box: {test_acc_bb}")
print(f"accuracy sTGMA: {test_acc_stgma}")
print(f"Fidelity: {test_fidel}")


#plt.gca().set_color_cycle(['red', 'green', 'blue', 'orange'])
plt.subplot(2, 2, 1)
plt.plot(save_loss1, color='red', alpha = 0.5)
plt.plot(save_share_loss1, color='blue', alpha = 0.5)
plt.title("Loss ANN")

plt.legend(['cross_entropy', 'share_loss1'], loc='upper left')

plt.subplot(2, 2, 2)
plt.plot(save_loss2)
plt.legend(['expected_loglikel'], loc='upper left')
plt.title("STGMA")

plt.subplot(2, 2, 3)
plt.plot(list_train_acc_bb, color='red', alpha = 0.4)
plt.plot(list_train_acc_stgma, color='blue', alpha = 0.3)
plt.plot(list_fidel_train, color='orange', alpha = 0.3)
plt.plot(list_nmi_train, color='green', alpha = 0.3)
plt.title("Training")
plt.ylim([0.4, 1.])
plt.legend(['train_acc_bb', 'train_acc_stgma', 'train fidelity','nmi_train'], loc='lower right')

plt.subplot(2, 2, 4)
plt.plot(list_test_acc_bb, color='red', alpha = 0.4)
plt.plot(list_test_acc_stgma, color='blue', alpha = 0.3)
plt.plot(list_fidel_test, color='orange', alpha = 0.3)
plt.plot(list_nmi_test, color='green', alpha = 0.3)
plt.title("Testing")
plt.legend(['test_acc_bb', 'test_acc_stgma', 'test fidelity', 'nmi_test'], loc='lower right')
plt.ylim([0.4, 1.])
plt.savefig(f'{directory}loss.png')
plt.close()

print("---Saving model---")
bb_directory = f"{directory}/bb_weights"
os.makedirs(bb_directory, exist_ok = True)

stgma_directory = f"{directory}/stgma_weights"
os.makedirs(stgma_directory, exist_ok = True)

tf.saved_model.save(model, stgma_directory)
tf.saved_model.save(black_box, bb_directory)