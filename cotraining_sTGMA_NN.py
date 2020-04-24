#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:18:12 2020

@author: gnanfack
"""


import numpy as np



import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp




from read_dataset_for_constraint import switch_dataset

from utils import plot_boundaries_hyperrect
from sTGMA import SoftTruncatedGaussianMixtureAnalysis

from black_box import BlackBoxNN

np.set_printoptions(precision=5)
tfd = tfp.distributions
np.random.seed(111)
tf.random.set_seed(111)



print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


n_components = 1

save_loss = []
#np.random.seed(903604963)
#np.random.seed(1595417368)
#seed = np.random.seed(159541736)
seed = np.random.seed(112)

np.random.seed(seed)
tf.random.set_seed(seed)








dataset_name = "data1"
type_eta = "eta_constant"
if_pca = False

X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map = \
    switch_dataset(dataset_name)(if_PCA = if_pca)
model = SoftTruncatedGaussianMixtureAnalysis(n_components = n_components, data_dim = X_train.shape[1],
											 n_classes = len(np.unique(y_train))

											 )


model.gmm_initialisation(X_train, y_train.astype(np.int32))

black_box = BlackBoxNN(nb_units = 128, nb_classes = len(np.unique(y_train)))
tf.print(black_box.trainable_variables)

optimizer_sTGMA = tf.optimizers.Adam(lr = 0.001)

optimizer_black_box = tf.optimizers.Adam(lr = 0.001)



@tf.function
def train_step_sTGMA(data, labels, responsibilities, eta, samples):

	model.eta.assign(eta)

	with tf.GradientTape() as tape:
		expected_loglikel = model(data, labels, responsibilities)
		share_loss = model.share_loss(X = samples,  black_box_model = black_box )
		loss = expected_loglikel
	gradients = tape.gradient(loss, model.trainable_variables)
	optimizer_sTGMA.apply_gradients(zip(gradients, model.trainable_variables))

	return expected_loglikel, share_loss, responsibilities, gradients


@tf.function
def train_step_black_box(data, labels_one_hot, samples):
	cross_ent = tf.keras.losses.CategoricalCrossentropy()
	with tf.GradientTape() as tape:

		share_loss = black_box.share_loss(X = samples,  sTGMA = model )
		cross_entropy = cross_ent(labels_one_hot, black_box(data))

		loss = cross_entropy #+ share_loss
	gradients = tape.gradient(loss , black_box.trainable_variables)

	optimizer_black_box.apply_gradients(zip(gradients, black_box.trainable_variables))

	return cross_entropy, share_loss, gradients


exp_log_lik_loss = {}
eta = 20
tol = 0.002
loss2 = 100.0
save_loss1 = []
save_loss2 = []
save_share_loss1 = []
save_share_loss2 = []
diff = []
directory = f"images_cotraining/datasets/{dataset_name}/boundaries"


# directory = f"{directory}/value_{eta}"

#os.makedirs(directory, exist_ok = True)
for i in range(50):
    #Expectation step
	if type_eta == "eta_variant":
		eta = (0.5)*np.sqrt(i) + 10



	filename = f"{directory}_{i}.png"
    #print(probs_x_given_k.shape)
	responsibilities = model.compute_responsibilities(X_train, y_train.astype(np.int32))
# 	plot_pdf_hyperrectangles(X_train, y_train.astype(np.int32), 0, 1, model.lower.numpy(), model.upper.numpy(),
#                           nb_hyperrectangles = model.n_components,
#                           file_name = f"{filename}_rectangles_{i}.png",
#                           color_map = color_map, mu = model.mu.numpy())

	    #print(responsabilities)
    #print(pi, probs_x_given_k)

    #Maximization
	print("Sampling ...")
	samples = model.sample_directly(nb_samples = 500).numpy()


	exp_log_lik_loss[i] = []

	#Learning black-box model
	for j in range(10):

		loss1, share_loss1, grad = train_step_black_box(data = X_train,
												  labels_one_hot = y_train_onehot, samples = samples)
		loss1, share_loss1 = loss1.numpy(), share_loss1.numpy()

	save_loss1.append(loss1)
	save_share_loss1.append(share_loss1)

	black_box_labels = np.argmax(black_box(X_train).numpy(), axis = 1)

	#Learning sTGMA
	for j in range(100):

		loss, share_loss2, resp, grad = train_step_sTGMA(data = X_train, labels = black_box_labels.astype(np.int32),
								responsibilities = responsibilities, eta = eta, samples = samples)
		loss, share_loss2, resp = loss.numpy(), share_loss2.numpy(), resp.numpy()
		exp_log_lik_loss[i].append(loss)


	plot_boundaries_hyperrect(X = X_train,
						   y = y_train,
						   x_axis= 0,
						   y_axis= 1,
						   black_box= black_box,
						   color_map= color_map,
						   file_name = filename,
						   sTGMA= model,
						   steps= 100)
	save_loss2.append(loss)
	save_share_loss2.append(share_loss2)



	if np.abs(loss2 - loss) < tol:
		break
	else:
		diff.append(loss2-loss)
		loss2 = loss
	print(f"***** Iteration {i} *****")
	print(f"loss1: {loss1}, loss2: {loss2}, shareloss: {share_loss2}")
#    plot_pdfR(X_train[:,0], X_train[:,1], f"{filename}_density_{i}.png", model, color_map)


