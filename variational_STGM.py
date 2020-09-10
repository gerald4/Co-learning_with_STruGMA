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

from utils import plot_hyperrectangles, plot_pdfR

np.set_printoptions(precision=5)
tfd = tfp.distributions
np.random.seed(111)
tf.random.set_seed(111)



dataset_name = "data1"
type_eta = "eta_constant"
if_pca = False

X_train, y_train, X_val, y_val, X_test, y_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler, color_map = \
    switch_dataset(dataset_name)(if_PCA = if_pca)


number_components = 2

save_loss = []
#np.random.seed(903604963)
#np.random.seed(1595417368)
#seed = np.random.seed(159541736)
seed = np.random.seed(112)

np.random.seed(seed)
tf.random.set_seed(seed)


class SoftTruncatedGaussianMixtureAnalysis(tf.Module):

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
        self.mu_scale = tf.Variable(
        	tf.ones((self.n_classes, self.n_components, 
        	self.data_dim)),
        	dtype = tf.float32, name="mu_scale"
        )


        self.lower = tf.Variable(
            np.random.randn(self.n_classes, self.n_components,
                            self.data_dim),
            dtype=tf.float32, name="lower"
        )

        self.lower_scale = tf.Variable(
        	 tf.ones((self.n_classes, self.n_components, 
        	self.data_dim)),
            dtype=tf.float32, name="lower_scale"
        	)


        self.upper = tf.Variable(
            np.random.randn(self.n_classes,
                            self.n_components, self.data_dim),
            dtype=tf.float32, name="upper"
        )

        self.upper_scale = tf.Variable(
        	 tf.ones((self.n_classes, self.n_components, 
        	self.data_dim)),
            dtype=tf.float32, name="upper_scale"
        	)


        self.sigma = tf.Variable(
            np.abs( np.random.randn(self.n_classes,
                            self.n_components,
                             self.data_dim)
                   ),
            dtype=tf.float32, name="sigma"
        )

		self.sigma_scale = tf.Variable(
		        	 tf.ones((self.n_classes, self.n_components, 
		        	self.data_dim)),
		            dtype=tf.float32, name="sigma_scale"
		        	)        

        self.logits_k = tf.Variable(
            np.random.randn(self.n_classes,
                   n_components),
            dtype = tf.float32,  name= "logits_k")

        self.logits_k_alpha = tf.Variable(
        	10.*tf.ones((self.n_classes,)),
        	dtype  = tf.float32, name= "logits_k_prior_variable"
        	)

        self.logits_y = tf.Variable(np.random.randn(self.n_classes
                                              ),
            dtype = tf.float32,  name= "logits_y", trainable = False)

        self.theta = tf.Variable(0.2, name = "smallest_margin", trainable = False)

        


    def gmm_initialisation(self, X_train, y_train):
        """This function intialises our STGMA using gaussian mixture model
        """

        self.n_points = X_train.shape[0]

        self.logits_q_z = tf.Variable(np.random.randn(self.n_points
                                              ),
            dtype = tf.float32,  name= "logits_q_z")

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

        self.p_mu = tfd.MultivariateNormalDiag(loc = tf.zeros((self.data_dim,)),
        									scale_diag = 10* tf.ones((self.data_dim,))
        									) 
        self.p_sigma = tfd.Independent( 
        	tfd.WishartTriL( 
        		df=5,
        		scale_tril=np.stack([np.eye(dims, dtype=dtype)]*self.n_components),
        	input_output_cholesky=True),
   			 reinterpreted_batch_ndims=1
   			 )

   		self.p_pi = tfd.Dirichlet( concentration = 10e-1 * tf.ones(self.n_components))


    @tf.function    
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

    @tf.function
    def log_likelihood(self, theta_samples,
    	z, mu, sigma, logits_k, lower, upper, X, y):

    	list_dist = [ 
    		tfd.Mixture(
          		cat = tfp.distributions.Categorical(logits = theta_samples[3][p]),
        		components = [tfd.Independent(tfd.Normal(loc = theta_samples[1][p,i,:],
                                              scale = np.finfo(np.float32).eps + tf.nn.softplus(theta_samples[2][p,i,:])),
                reinterpreted_batch_ndims=1) for i in range(self.n_components)
          ])
    		for p in range(theta_samples[0].shape[0])
        	]

        log_pu = self.p_mu.logprob(theta_samples[1])

        log_psigma = self.p_sigma.logpob(theta_samples[2])

        log_plogits = self.p_pi.logprob(theta_samples[3])


        #Shape [K x N x M]

        list_p_more_lower_given_x = [
        								tf.stack([X-tf.expand_dims(theta_samples[4][p,i,:],
                                                          axis=0) for i in range(self.n_components)])
        								for p in range(theta_samples[0].shape[0])
        							
        							]
        #Shape [K x N x M]
        list_p_less_upper_given_x = [
        								tf.stack([tf.expand_dims(theta_samples[5][p,i,:],
                                                        axis=0) - X for i in range(self.n_components)])
        								for p in range(theta_samples[0].shape[0])
        							]

        log_p_more_lower_given_x = ( tf.transpose(tf.reduce_sum( - tf.nn.softplus(-self.eta *
                                                                    p_more_lower_given_x)
                                                                , axis = -1))
                                   )

        log_p_less_upper_given_x = (tf.transpose(tf.reduce_sum( - tf.nn.softplus(-self.eta *
                                                                    p_less_upper_given_x)
                                                        , axis = -1))
        							)


	 @tf.function    
	    def normalizing_constant(self, way = "independent", mu, sigma):
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
