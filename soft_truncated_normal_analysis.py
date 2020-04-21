#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:18:12 2020

@author: gnanfack
"""

import os
import numpy as np
from time import time
import pandas as pd
from scipy.special import softmax
from scipy.special import expit
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import matplotlib.cm as cm


from read_dataset_for_constraint import switch_dataset

from utils import plot_hyperrectangles, plot_pdfR, plot_pdf_hyperrectangles

np.set_printoptions(precision=5)
tfd = tfp.distributions
np.random.seed(111)
tf.random.set_seed(111)



print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


n_components = 2

save_loss = []
#np.random.seed(903604963)
#np.random.seed(1595417368)
#seed = np.random.seed(159541736)
seed = np.random.seed(112)

np.random.seed(seed)
tf.random.set_seed(seed)



class SoftTruncatedGaussianMixtureAnalysis(tf.keras.Model):

	def __init__(self, n_components, data_dim, n_classes, seed = 111):
		"""
		This function creates the variables of the model.
		n_components: number of components for each mixture per class
		data_dim: number of features
		n_classes: number of classes
		"""

		super(SoftTruncatedGaussianMixtureAnalysis, self).__init__()
		np.random.seed(seed)
		tf.random.set_seed(seed)

		#Value eta for logisitic
		self.eta = tf.Variable(0.0, trainable=False)

		self.n_components = n_components
		self.stable = tf.constant(np.finfo(np.float32).eps)

		self.data_dim = data_dim
		self.n_classes = n_classes

		self.mu = tf.Variable(
			np.random.randn(self.n_classes, self.n_components,
				   self.data_dim),
			dtype=tf.float32, name="mu"
        )

		self.lower = tf.Variable(
            np.random.randn(self.n_classes, self.n_components,
							self.data_dim),
			dtype=tf.float32, name="lower"
        )


		self.upper = tf.Variable(
            np.random.randn(self.n_classes,
							self.n_components, self.data_dim),
			dtype=tf.float32, name="upper"
        )

		self.sigma = tf.Variable(
            np.abs( np.random.randn(self.n_classes,
							self.n_components,
							 self.data_dim)
				   ),
			dtype=tf.float32, name="sigma"
        )
		self.logits_k = tf.Variable(
			np.random.randn(self.n_classes,
				   n_components),
			dtype = tf.float32,  name= "logits_k")

		self.logits_y = tf.Variable(np.random.randn(self.n_classes
											  ),
			dtype = tf.float32,  name= "logits_y", trainable = False)


	def gmm_initialisation(self, X_train, y_train):
		"""This function intialises our STGMA using gaussian mixture model
		"""

        #Split X_train by unique Y_train
		y_unique = np.unique(y_train)
		#For each unique Y_train, create a GMM and initialise parameters of our STGMA
		for i in range(len(y_unique)):
			gmm = GaussianMixture(n_components = self.n_components,
							   covariance_type="diag")
			gmm.fit(X_train[np.where(y_train == y_unique[i])[0]])
			low = gmm.means_ - np.sqrt(gmm.covariances_)
			upp = gmm.means_ + 0.4*np.sqrt(gmm.covariances_)

			self.lower[i].assign(low.astype(np.float32))
			self.upper[i].assign(upp.astype(np.float32))
			self.mu[i].assign(gmm.means_.astype(np.float32))

			self.sigma[i].assign(gmm.covariances_.astype(np.float32))

			self.logits_y[i].assign(X_train[np.where(y_train == y_unique[i])[0]].shape[0]/X_train.shape[0])

		self.y_unique = y_unique


	def normalizing_constant(self, way = "independent"):
		"""This function computes the normalizing constant which envolves the integral
        """
		if way == "independent":
            # Reconstructing the independent variables
			dict1 = {
                (c, k, d): tfd.Normal(
                    loc=self.mu[c, k, d],
                    scale=np.finfo(np.float32).eps + tf.nn.softplus(self.sigma[c, k, d]),
                )
				for c in range(self.n_classes)
                for d in range(self.data_dim)
                for k in range(self.n_components)
            }

            #Return \int_{lower}^{upper} Normal(x;\mu, \Sigma) dx1 ... dxn
			return (
					tf.reduce_prod(tf.stack([ [ [ dict1[(c,k,d)].cdf(self.upper[c,k,d])
												  - dict1[(c,k,d)].cdf(self.lower[c,k,d])
												  for d in range(self.data_dim)
												  ] for k in range(self.n_components)
											  ] for c in range(self.n_classes)
											 ]
											),
						axis = - 1, keepdims = True) + self.stable
				  )


	def compute_log_pdf(self, X, y):
		dist = tfd.Mixture(
          cat = tfp.distributions.Categorical(logits = self.logits_k[y]),
        components = [tfd.Independent(tfd.Normal(loc = self.mu[y,i,:],
                                              scale = np.finfo(np.float32).eps + tf.nn.softplus(self.sigma[y,i,:])),
                              reinterpreted_batch_ndims=1) for i in range(self.n_components)
          ])


        #Shape [K x N x M]
		p_more_lower_given_x = tf.stack([X-tf.expand_dims(self.lower[y,i,:],
                                                          axis=0) for i in range(self.n_components)])
        #Shape [K x N x M]
		p_less_upper_given_x = tf.stack([tf.expand_dims(self.upper[y,i,:],
                                                        axis=0) - X for i in range(self.n_components)])

		log_p_more_lower_given_x = ( tf.transpose(tf.reduce_sum( - tf.nn.softplus(-self.eta *
                                                                    p_more_lower_given_x)
                                                                , axis = -1))
                                   )

		log_p_less_upper_given_x = (tf.transpose(tf.reduce_sum( - tf.nn.softplus(-self.eta *
                                                                    p_less_upper_given_x)
                                                        , axis = -1))
                                   )

		return (
				tf.expand_dims(dist._cat_probs(log_probs=True), axis = 0)
                - tf.math.log(tf.transpose(self.normalizing_constant()[y]
										   )
							  )
                + log_p_more_lower_given_x
                + log_p_less_upper_given_x
                + tf.transpose(tf.stack([dist.components[i].log_prob(X)
                                                                    for i in range(self.n_components)
										]
										)
							   )
               )

	def log_joint_prob(self, X):
		return tf.transpose (
			tf.stack(
				[
				tf.reduce_logsumexp(self.compute_log_pdf(X, c), axis = -1) + tf.math.log(self.logits_y[c])
				for c in self.y_unique
				]
			)
		)
	def log_pdf(self, X):
		return tf.reduce_logsumexp(self.log_joint_prob(X), axis = -1)


	def compute_log_conditional_distribution(self, X):
		log_joint_prob = tf.transpose (
			tf.stack(
				[
				tf.reduce_logsumexp(self.compute_log_pdf(X, c), axis = -1) + tf.math.log(self.logits_y[c])
				for c in self.y_unique
				]
			)
		)

		return log_joint_prob - tf.reduce_logsumexp(log_joint_prob, axis = -1, keepdims = True)

	def share_loss(self, X, black_box_model):
		kl = tf.keras.losses.KLDivergence(
		reduction=tf.keras.losses.Reduction.SUM)


		return tfp.monte_carlo.expectation(
			f=lambda x: kl(
				tf.exp(
					self.compute_log_conditional_distribution(x)
					),
				black_box_model.predict(x)
				),
			samples = X,
			log_prob = model.log_pdf,
			   use_reparameterization= False
		)



	def predict(self, X):

		cond_prob = tf.exp(
			self.compute_log_conditional_distribution(
				X
				)
			)
		return np.argmax(
			cond_prob.numpy(),
			axis = -1
			)


	@tf.function
	def __call__(self, X, y, responsabilities):

		#Compute expected likelihood per class


		list_likelihood = []
		#For each unique y_train, create a compute the expected log-likelihood
		for i in range(len(self.y_unique)):
			list_likelihood.append(
				tf.reduce_mean(
					tf.multiply(
						tf.gather_nd(responsabilities[self.y_unique[i]],
									tf.where(tf.equal(y, self.y_unique[i]))
									),
						self.compute_log_pdf(
							tf.gather_nd(X, tf.where(tf.equal(y, self.y_unique[i]))),
							self.y_unique[i]
							)
						)
					)
				)




        #New losss:
		return - tf.reduce_sum(list_likelihood)




def compute_responsibilities(X, y, model):

	responsibilities = np.array(
		np.zeros(
			shape=(model.n_classes, X.shape[0], model.n_components)
			).astype(np.float32)
		)

	y_unique = np.unique(y)

	for c in y_unique:
		responsibilities[c] = softmax(model.compute_log_pdf(X, c), axis = 1)


	return responsibilities


print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


dataset_name = "data1"
type_eta = "eta_constant"
if_pca = False

X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map = \
    switch_dataset(dataset_name)(if_PCA = if_pca)
model = SoftTruncatedGaussianMixtureAnalysis(n_components = 2, data_dim = X_train.shape[1],
											 n_classes = len(np.unique(y_train))
											 )
model.gmm_initialisation(X_train, y_train.astype(np.int32))

optimizer = tf.optimizers.Adam(lr = 0.001)

@tf.function
def train_step(data, labels, responsibilities, eta):
	model.eta.assign(eta)
	#tf.print("toto", labels)
	with tf.GradientTape() as tape:
		current_loss = model(data, labels, responsibilities)

	gradients = tape.gradient(current_loss, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return current_loss, responsibilities, gradients


lloss= {}
eta = 20
tol = 0.002
loss1 = 100.0
diff = []
directory = f"./images_sTGMA/datasets/{dataset_name}/{type_eta}"


directory = f"{directory}/value_{eta}"

os.makedirs(directory, exist_ok = True)
for i in range(50):
    #Expectation step
	if type_eta == "eta_variant":
		eta = (0.5)*np.sqrt(i) + 10



	filename = f"{directory}/image"
    #print(probs_x_given_k.shape)
	responsibilities = compute_responsibilities(X_train, y_train.astype(np.int32), model)
# 	plot_pdf_hyperrectangles(X_train, y_train.astype(np.int32), 0, 1, model.lower.numpy(), model.upper.numpy(),
#                           nb_hyperrectangles = model.n_components,
#                           file_name = f"{filename}_rectangles_{i}.png",
#                           color_map = color_map, mu = model.mu.numpy())

	    #print(responsabilities)
    #print(pi, probs_x_given_k)

    #Maximization
	lloss[i] = []

	for j in range(100):

		loss, resp, grad = train_step(data = X_train, labels = y_train.astype(np.int32),
								responsibilities = responsibilities, eta = eta)
		loss, resp = loss.numpy(), resp.numpy()
		lloss[i].append(loss)

	save_loss.append(loss)

	if np.abs(loss1 - loss) < tol:
		break
	else:
		diff.append(loss1-loss)
		loss1 = loss
	print(f"Iteration: {i}, loss: {loss}")
#    plot_pdfR(X_train[:,0], X_train[:,1], f"{filename}_density_{i}.png", model, color_map)


