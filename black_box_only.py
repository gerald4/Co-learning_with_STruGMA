

import numpy as np
import os
from time import time
import datetime
import gc

from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorboard.plugins.hparams import api as hp

from read_dataset_for_constraint import switch_dataset

from utils import plot_boundaries_hyperrect
from sTGMA import SoftTruncatedGaussianMixtureAnalysis

from black_box import BlackBoxNN
from config import  config_params, hyper_params



@tf.function
def train_step_black_box(data, labels_one_hot, samples, weights = None, _lambda = 1.):
    cross_ent = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as tape:

        
        cross_entropy = cross_ent(labels_one_hot, black_box(data))

        loss = cross_entropy + black_box.losses()
    gradients = tape.gradient(loss , black_box.trainable_variables)

    optimizer_black_box.apply_gradients(zip(gradients, black_box.trainable_variables))

    return cross_entropy #, share_loss #, gradients


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

    X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map, _ = \
        switch_dataset(dataset_name)(if_PCA = False)


    black_box = BlackBoxNN(nb_units = hyper_params["nb_units"], nb_classes =len(np.unique(y_train)))
    # tf.print(black_box.trainable_variables)


    optimizer_black_box = tf.optimizers.Adam(lr = hyper_params["stgma_lr"])

    #exp_log_lik_loss = {}
    eta = hyper_params["value_eta"]
    tol = hyper_params["tol"]
    loss2 = 100000.0
    save_loss1 = []
 
    list_train_acc_bb = []


    list_test_acc_bb = []


    metrics_train = ['acc_nn', 
                'cross_entropy_vs_shareloss1']

    # metrics_test = ['acc_nn_test', 'acc_stgma_test',
    # 'fidel_test', 'share_loss1',  'share_loss2'
    # ]

    diff = []
    directory = f"images_black_box/datasets/{dataset_name}/components_{n_components}/value_{str(_lambda)}"

    if converge_bb_before:
        if config_params["weights"]:
            directory = f"{directory}∕converge_bbb_before_with_weights"
        else:
            directory = f"{directory}∕converge_bbb_before"
    elif config_params["weights"]:
        directory = f"{directory}∕with_weights"



    os.makedirs(directory, exist_ok = True)
    directory = f"{directory}/"

    log_dir = f"{directory}/logs"
    os.makedirs(log_dir, exist_ok = True)

    # writer_train = tf.summary.create_file_writer(f"{log_dir}/train")
    # writer_test = tf.summary.create_file_writer(f"{log_dir}/test")
    writer_train = tf.summary.create_file_writer(f"{log_dir}/train")
    writer_test = tf.summary.create_file_writer(f"{log_dir}/test")


    # directory = f"{directory}/value_{eta}"



    for step in range(hyper_params["global_steps"]):
        #Expectation step
        if type_eta == "eta_variant":
            eta = (0.5)*np.sqrt(step) + eta



        filename = f"{directory}boundaries_importance_{step}"


        #print(weights)

        #exp_log_lik_loss[step] = []
        
        #Learning black-box model
        toc_tic = time()


        for j in range(hyper_params["bb_steps"]):
            print(f"Black-box update iteration {j}")
            loss1 = train_step_black_box(data = X_train,
                                                      labels_one_hot = y_train_onehot, samples = samples,
                                                      weights = weights,
                                                      _lambda = _lambda)
            loss1 = loss1.numpy()#, share_loss1.numpy()

        save_loss1.append(loss1)
        #save_share_loss1.append(share_loss1)

        black_box_probs = black_box(X_train)
        black_box_labels = np.argmax(black_box_probs.numpy(), axis = 1)

        if not(config_params["weights"]):
            black_box_probs = tf.one_hot(black_box_labels, len(np.unique(y_train)))



        tac_tic = time()

        print(f"Time for iteration {step}: {tac_tic - toc_tic} ")

        #print("---->Projection<----")
        #model.projection(X_train, responsibilities)
        #print(model.no_ovelap_test())
        #print(tf.where(tf.less(model.no_ovelap_test(),0)))
        #print(model.upper- model.lower)
            #exp_log_lik_loss[step].append(loss)



        y_train_bb = np.argmax(black_box.predict(X_train).numpy(), axis = 1)
        y_test_bb = np.argmax(black_box.predict(X_test).numpy(), axis = 1)


        train_acc_bb = accuracy_score(y_train, y_train_bb)

        test_acc_bb = accuracy_score(y_test, y_test_bb)



        list_train_acc_bb.append(train_acc_bb)

        list_test_acc_bb.append(test_acc_bb)

        values_train = [train_acc_bb,
                loss1]
        values_test = [test_acc_bb] 
        
        write_metrics(writer_test, metrics_train[:-1], tf.Variable(values_test, trainable = False), tf.constant(step, tf.int64))
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



        print(f"loss: {loss}")
    #    plot_pdfR(X_train[:,0], X_train[:,1], f"{filename}_density_{i}.png", model, color_map)

        gc.collect()


print('----------Training----------')
print(f"accuracy black box: {train_acc_bb}")


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
plt.title("Training")
plt.ylim([0.4, 1.])
plt.legend(['train_acc_bb', 'train_acc_stgma', 'train fidelity'], loc='lower right')

plt.subplot(2, 2, 4)
plt.plot(list_test_acc_bb, color='red', alpha = 0.4)
plt.plot(list_test_acc_stgma, color='blue', alpha = 0.3)
plt.plot(list_fidel_test, color='orange', alpha = 0.3)
plt.title("Testing")
plt.legend(['test_acc_bb', 'test_acc_stgma', 'test fidelity'], loc='lower right')
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