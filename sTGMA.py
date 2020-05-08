#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:36:44 2020

@author: gnanfack
"""

import numpy as np
from time import time

from sklearn.mixture import GaussianMixture


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import matplotlib.cm as cm


from read_dataset_for_constraint import switch_dataset

from utils import plot_hyperrectangles, plot_pdfR, plot_pdf_hyperrectangles

tfd = tfp.distributions


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

        self.theta = tf.Variable(0.2, name = "smallest_margin", trainable = False)


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

    def sample_per_class(self, nb_samples):

        def run_chain(num_results, num_burnin_steps, current_state, kernel_type):
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
            num_results = num_results,
            num_burnin_steps=num_burnin_steps,
            current_state = current_state,
            kernel=kernel_type,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            return is_accepted, samples


        list_samples = []
        tic = time()

        for i in range(len(self.y_unique)):
            print(f"Sampling class: {self.y_unique[i]}")
            num_results = int(self.logits_y[i].numpy()*nb_samples)
            num_burnin_steps = int(1e3)
            adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn= lambda x: tf.reduce_logsumexp(
                        self.compute_log_pdf(x, self.y_unique[i]
                        ),
                         axis = -1
                         ),
                    num_leapfrog_steps=3,
                    step_size= 1.,
                    state_gradients_are_stopped=True),
                num_adaptation_steps=int(num_burnin_steps * 0.8))
            acceptance, samples = run_chain(
                num_results,
                num_burnin_steps,
                tf.Variable([self.lower[i,0]/2 + self.upper[i,0]/2], trainable = False),
                 adaptive_hmc
                 )

            list_samples.append(tf.squeeze(samples))
        tac = time()
        print(f"Time for sampling: {(tac-tic)/60} min")
        return list_samples


    def sample_importance(self, nb_samples):
        list_samples = []
        list_weights = []
        tic = time()
        for i in range(len(self.y_unique)):
            n_samples = int(self.logits_y[i].numpy()*nb_samples)
            dist = tfd.Mixture( cat = tfp.distributions.Categorical(
                 logits = self.logits_k[i]),
                 components = [
                     tfd.Independent(
                         tfd.TruncatedNormal(loc = self.mu[i,j,:],
                            low = self.lower[i,j,:],
                            high = self.upper[i,j,:],
                            scale = np.finfo(np.float32).eps + tf.nn.softplus(
                                self.sigma[i,j,:])
                            ),
                         reinterpreted_batch_ndims=1) for j in range(
                             self.n_components
                             )
                             ]
                             )
            samples = dist.sample(n_samples)
            weights = (
                tf.math.exp(tf.reduce_logsumexp(self.compute_log_pdf(samples, i), axis = -1))
                    /
                tf.math.exp(dist.log_prob(samples))
                )
            list_samples.append(samples)
            list_weights.append(weights)
        tac = time()
        print(f"Time for sampling: {(tac-tic)/60} min")

        return tf.concat(list_samples, axis = 0), tf.concat(list_weights, axis = 0)



    def sample_directly(self, nb_samples):
        @tf.function #(experimental_compile=True)
        def run_chain(num_results, num_burnin_steps, current_state, kernel_type):
            # Run the chain (with burn-in).
            samples, is_accepted = tfp.mcmc.sample_chain(
            num_results = num_results,
            num_burnin_steps=num_burnin_steps,
            current_state = current_state,
            kernel=kernel_type,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

            is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
            return is_accepted, samples


        tic = time()
        print("Start sampling")
        num_results = nb_samples
        num_burnin_steps = int(1e3)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn= self.log_pdf,
                    num_leapfrog_steps=3,
                    step_size= 1.,
                    state_gradients_are_stopped=True),
                num_adaptation_steps=int(num_burnin_steps * 0.8))
        acceptance, samples = run_chain(
                num_results,
                num_burnin_steps,
                tf.Variable([tf.reduce_mean(self.mu, axis = [0,1])], trainable = False),
                 adaptive_hmc
                 )

        samples = tf.squeeze(samples)
        tac = time()
        print(f"Time for sampling: {(tac-tic)/60} min")
        return samples

    def log_joint_prob(self, X):
        return tf.transpose (
            tf.stack(
                [
                tf.reduce_logsumexp(self.compute_log_pdf(X, c), axis = -1) + tf.math.log(self.logits_y[c] + self.stable)
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
                tf.reduce_logsumexp(self.compute_log_pdf(X, c), axis = -1) + tf.math.log(self.logits_y[c] + self.stable)
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
                black_box_model(x)
                ),
            samples = X,
            log_prob = self.log_pdf,
               use_reparametrization= False
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




    def compute_responsibilities(self, X, y):

        responsibilities = np.array(
            np.zeros(
                shape=(self.n_classes, X.shape[0], self.n_components)
                ).astype(np.float32)
            )

        #y_unique = np.unique(y)

        for c in self.y_unique:
            responsibilities[c] = tf.nn.softmax(self.compute_log_pdf(X, c), axis = 1).numpy()


        return responsibilities