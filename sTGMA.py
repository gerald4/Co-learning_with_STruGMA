#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:36:44 2020

@author: gnanfack
"""

import numpy as np
from time import time

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import matplotlib.cm as cm


from read_dataset_for_constraint import switch_dataset

from utils import plot_hyperrectangles, plot_pdfR, plot_pdf_hyperrectangles
from truncated_normal import SoftTruncatedNormal

tfd = tfp.distributions


class SoftTruncatedGaussianMixtureAnalysis(tf.Module):

    def __init__(self, n_components, data_dim, n_classes, theta, seed = 111, m_max_min = 10.):
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
        self.eta = tf.Variable(20.0, trainable=False)

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

        self.m_max_min = tf.constant(10., name = "m_max_min")


    def gmm_initialisation(self, X_train, y_train):
        """This function intialises our STGMA using gaussian mixture model
        """

        #Split X_train by unique Y_train
        y_unique = np.unique(y_train)
        #For each unique Y_train, create a GMM and initialise parameters of our STGMA
        for i in range(len(y_unique)):
            gmm = GaussianMixture(n_components = self.n_components,
                               covariance_type="diag", reg_covar=1e-05)
            gmm.fit(X_train[np.where(y_train == y_unique[i])[0]])
            low = gmm.means_ - 0.2*np.sqrt(gmm.covariances_)
            upp = gmm.means_ + 0.4*np.sqrt(gmm.covariances_)

            self.lower[i].assign(low.astype(np.float32))
            self.upper[i].assign(upp.astype(np.float32))
            self.mu[i].assign(gmm.means_.astype(np.float32))

            self.sigma[i].assign(gmm.covariances_.astype(np.float32))

            self.logits_y[i].assign(X_train[np.where(y_train == y_unique[i])[0]].shape[0]/X_train.shape[0])
            #print(f"-----components ---->>>>> {np.sum(model.weights_ > 0.01)}")
        self.y_unique = y_unique

    #@tf.function    
    def normalizing_constant(self, way = "independent"):
        """This function computes the normalizing constant which envolves the integral
        """
        #print("---tracing-normalising")
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

    #@tf.function
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

    #@tf.function
    def projection(self, X, y, resp, weights, t_range):

        #print("toooooooooooooooooooooooooooooooooooooooo")

        #@tf.function
        def expec_ll(alpha1, alpha2):#, i, j, c1, c2):

            temp_lower = tf.identity(self.lower)
            temp_upper = tf.identity(self.upper)

            self.lower.assign(alpha1)
            self.upper.assign(alpha2)

            log_cond = self.expected_ll(X, y, resp, weights)

            self.lower.assign(temp_lower)
            self.upper.assign(temp_upper)

            #tf.print(self.lower)
            #tf.print(self.upper)

            return log_cond

        # def empirical_entropy(X, alpha1, alpha2, i, j, c1, c2):

        #     temp_lower = tf.identity(self.lower)
        #     temp_upper = tf.identity(self.upper)

        #     self.lower.assign(alpha1)
        #     self.upper.assign(alpha2)

        #     log_cond = self.compute_log_conditional_distribution(X)

        #     self.lower.assign(temp_lower)
        #     self.upper.assign(temp_upper)

        #     #tf.print(self.lower)
        #     #tf.print(self.upper)

        #     return - tf.reduce_sum(tf.expand_dims(resp[c1,..., i] + resp[c2,...,j], axis = -1)*tf.exp(log_cond)*log_cond)

        # index_i = tf.TensorArray(dtype =tf.int32, size = 0, dynamic_size = True)
        # index_j = tf.TensorArray(dtype = tf.int32, size = 0, dynamic_size =True)
        # tf.print(self.no_ovelap_test(), output_stream="file:///tmp/tensor.txt")


        #tf.print("toooooooooooooooooooooooooooooooooooooooo")

        tmp_indexes = tf.where(tf.less(self.no_ovelap_test(), -self.theta/50.))

        #tf.print(tmp_indexes, output_stream="file:///tmp/tensor2.txt")

        #tf.print(tf.size(tmp_indexes))
        #print("toto")
        #tf.print(t_range)

        while not(tf.equal(tf.size(tmp_indexes), 0)):
            #print("while  looop")


            classes = tf.cast(tf.math.floordiv(tmp_indexes, self.n_components), tf.int32) 
            good_indexes = tf.cast(tf.math.floormod(tmp_indexes, self.n_components), tf.int32)

            score = tf.TensorArray(dtype =tf.float32, size = 0, dynamic_size = True, name = "score",
                clear_after_read=False)
            #Matrix of updates
            alpha1 = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True, name = "alpha1",
                clear_after_read=False)

            alpha2 = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True, name = "alpha2",
                clear_after_read=False)

            #tf.print(classes)
            #tf.print(good_indexes)
                
            #For each update, compute the entropy
        

            for it in tf.range(tf.minimum(tf.constant(self.data_dim), 20)):
                #print("toto")
                d = t_range[it]
                #tf.print(self.lower)
                #print("d loooooop")

                if self.upper[classes[0,0],good_indexes[0,0],d] >  self.upper[classes[0,1],good_indexes[0,1],d]:

                    alpha1 = alpha1.write(2*it, tf.tensor_scatter_nd_update(self.lower, 
                                    [[classes[0,0],good_indexes[0,0],d]],
                                [self.upper[classes[0,1], good_indexes[0,1],d]] ))

                    alpha2 = alpha2.write(2*it, self.upper)
                    score = score.write(2*it, expec_ll(alpha1.read(2*it), alpha2.read(2*it)))
                        #,good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))

                else: 

                    alpha1 = alpha1.write(2*it, tf.tensor_scatter_nd_update(self.lower, 
                    [[classes[0,1],good_indexes[0,1],d]],
                    [ self.upper[classes[0,0], good_indexes[0,0],d]]))

                    alpha2 = alpha2.write(2*it, self.upper)

                    score = score.write(2*it , expec_ll(alpha1.read(2*it), alpha2.read(2*it )))
                       #,
                        #good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))

                if self.lower[classes[0,0], good_indexes[0,0],d] < self.lower[classes[0,1], good_indexes[0,1], d]:

                    alpha2 = alpha2.write(2*it+1, tf.tensor_scatter_nd_update(self.upper, 
                    [[classes[0,0],good_indexes[0,0],d]],
                    [ self.lower[classes[0,1], good_indexes[0,1],d]] ))

                    alpha1 = alpha1.write(2*it+1, self.lower)

                    score = score.write(2*it + 1, expec_ll(alpha1.read(2*it + 1), alpha2.read(2*it + 1)))
                    #,
                     #   good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))

                else:

                    alpha2 = alpha2.write(2*it+1, tf.tensor_scatter_nd_update(self.upper, 
                    [[classes[0,1],good_indexes[0,1],d]],
                    [ self.lower[classes[0,0], good_indexes[0,0],d]] ))

                    alpha1 = alpha1.write(2*it+1, self.lower)

                    score = score.write(2*it + 1, expec_ll(alpha1.read(2*it + 1), alpha2.read(2*it + 1)))
                    #,
                     #   good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))

                # alpha1 = alpha1.write(4*d, tf.tensor_scatter_nd_update(self.lower, 
                #     [[classes[0,0],good_indexes[0,0],d]],
                #     [tf.math.maximum(self.lower[classes[0,0],good_indexes[0,0],d],
                #         tf.math.minimum(self.upper[classes[0,0], good_indexes[0,0],d],
                #             self.upper[classes[0,1], good_indexes[0,1],d]))] ))
                # #tf.print(d)
                # alpha2 = alpha2.write(4*d, self.upper)
                # #tf.print("totototoototo")
                # #emp_ent = empirical_entropy()
                # #tf.print(alpha1.read(4*d))
                # score = score.write(4*d, empirical_entropy(X, alpha1.read(4*d), alpha2.read(4*d),
                #     good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))

                # #tf.print(alpha1.read(4*d))

                # #tf.print(self.lower)

                # alpha1 = alpha1.write(4*d+1, tf.tensor_scatter_nd_update(self.lower, 
                #     [[classes[0,1],good_indexes[0,1],d]],
                #     [tf.math.maximum(self.lower[classes[0,1],good_indexes[0,1],d],
                #         tf.math.minimum(self.upper[classes[0,1], good_indexes[0,1],d],
                #             self.upper[classes[0,0], good_indexes[0,0],d]))]))

                # alpha2 = alpha2.write(4*d+1, self.upper)

                # score = score.write(4*d + 1, empirical_entropy(X, alpha1.read(4*d +1 ), alpha2.read(4*d + 1),
                #     good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))
                # #tf.print(self.lower)

                # alpha2 = alpha2.write(4*d+2, tf.tensor_scatter_nd_update(self.upper, 
                #     [[classes[0,0],good_indexes[0,0],d]],
                #     [tf.math.minimum(self.upper[classes[0,0],good_indexes[0,0],d],
                #         tf.math.maximum(self.lower[classes[0,0], good_indexes[0,0],d],
                #             self.lower[classes[0,1], good_indexes[0,1],d]))] ))

                # alpha1 = alpha1.write(4*d+2, self.lower)

                # score = score.write(4*d + 2, empirical_entropy(X, alpha1.read(4*d + 2), alpha2.read(4*d + 2),
                #     good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))

                # #tf.print(self.lower)

                # alpha2 = alpha2.write(4*d+3, tf.tensor_scatter_nd_update(self.upper, 
                #     [[classes[0,1],good_indexes[0,1],d]],
                #     [tf.math.minimum(self.upper[classes[0,1],good_indexes[0,1],d],
                #         tf.math.maximum(self.lower[classes[0,1], good_indexes[0,1],d],
                #             self.lower[classes[0,0], good_indexes[0,0],d]))] ))

                # alpha1 = alpha1.write(4*d+3, self.lower)

                # score = score.write(4*d + 3, empirical_entropy(X, alpha1.read(4*d + 3), alpha2.read(4*d + 3),
                #     good_indexes[0,0], good_indexes[0,1], classes[0,0], classes [0,1]))

                #tf.print(self.lower)



            #tf.print(score.stack())

            #change the values of alpha corresponding to the lowest update
            true_score = score.stack()
            #ind = tf.cast(tf.math.argmin(tf.boolean_mask(true_score, tf.greater(true_score,0))), tf.int32)
            ind = tf.cast(tf.math.argmin(true_score), tf.int32)
            #tf.print(ind)

            self.lower.assign(alpha1.read(ind))
            self.upper.assign(alpha2.read(ind))    

            #Re-compute the no-overlapp    
            tmp_indexes = tf.where(tf.less(self.no_ovelap_test(), - self.theta))
            #if not(tf.equal(tf.size(tmp_indexes), 0)):
            #    tf.print(self.no_ovelap_test()[tmp_indexes[0,0], tmp_indexes[0,1]])
            tf.print("number of remaining overlapping: ", tf.size(tmp_indexes))
            print("number of remaining overlapping: ", tf.size(tmp_indexes))


            #print("voiciiiiiii")

        #tf.print(tmp_indexes)

    #@tf.function
    def no_ovelap_test(self):
        low = tf.reshape(self.lower, (self.lower.shape[0]*self.lower.shape[1],)+ (self.lower.shape[2:]))
        
        upp = tf.reshape(self.upper, (self.upper.shape[0]*self.upper.shape[1],)+ (self.upper.shape[2:]))
        
        centers = (1./2)*(upp + low)
        
        lenghts = (1./2)*(upp - low)
        
        pairwise_centers = tf.abs(centers[tf.newaxis, ...] - centers[:, tf.newaxis,...])
        
        pairwise_lengths = lenghts[tf.newaxis, ...] + lenghts[:, tf.newaxis, ...]
        
        no_overlap_mat = tf.reduce_max(pairwise_centers - pairwise_lengths, axis=-1)

        return tf.linalg.band_part(no_overlap_mat, -1, 0) - tf.linalg.band_part(no_overlap_mat, 0, 0)



    #@tf.function
    def sample_importance(self, nb_samples):
        list_samples = []
        list_weights = []
        tic = time()
        for i in range(len(self.y_unique)):
            n_samples = tf.dtypes.cast(self.logits_y[i]*nb_samples, tf.int32)
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

        return tf.concat(list_samples, axis = 0) #, tf.concat(list_weights, axis = 0)



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

    #@tf.function
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
        return tf.argmax(
            cond_prob,
            axis = -1
            )

    #@tf.function
    def expected_ll (self, X, y, responsibilities, weights):

        list_likelihood = tf.TensorArray(dtype =tf.float32, size =0, dynamic_size= True)
        #For each unique y_train, create a compute the expected log-likelihood
        for i in range(len(self.y_unique)):
            list_likelihood = list_likelihood.write( i, 
                tf.reduce_mean(
                    tf.multiply(
                        tf.multiply(weights[:,self.y_unique[i], tf.newaxis],
                                responsibilities[self.y_unique[i]]
                            ),
                        # tf.gather_nd(responsabilities[self.y_unique[i]],
                        #             tf.where(tf.equal(y, self.y_unique[i]))
                        #             ),
                        self.compute_log_pdf(
                            X, 
                            self.y_unique[i]
                            # tf.gather_nd(X, tf.where(tf.equal(y, self.y_unique[i]))),
                            # self.y_unique[i]
                            )
                        )
                    )
                )

        return - tf.reduce_sum(list_likelihood.stack())


    #@tf.function
    def __call__(self, X, y, responsabilities, weights):

        #Compute expected likelihood per class


        list_likelihood = []
        #For each unique y_train, create a compute the expected log-likelihood
        for i in range(len(self.y_unique)):
            list_likelihood.append(
                tf.reduce_mean(
                    tf.multiply(
                        tf.multiply(weights[:,self.y_unique[i], tf.newaxis],
                                responsabilities[self.y_unique[i]]
                            ),
                        # tf.gather_nd(responsabilities[self.y_unique[i]],
                        #             tf.where(tf.equal(y, self.y_unique[i]))
                        #             ),
                        self.compute_log_pdf(
                            X, 
                            self.y_unique[i]
                            # tf.gather_nd(X, tf.where(tf.equal(y, self.y_unique[i]))),
                            # self.y_unique[i]
                            )
                        )
                    )
                )


        #@tf.function
        def noverlap():
            """This function is not important yet
            It helps to penalize overalapping rectangles"""
    
            centers = (1./2)*(self.lower + self.upper)
            radii = tf.norm((1./2)*(self.upper - self.lower) , axis= 1, keepdims=True)

            r = tf.reduce_sum(centers*centers, 1)

                # Turning r into vector
            r = tf.reshape(r, [-1, 1])
            D = r - 2*tf.matmul(centers, tf.transpose(centers)) + tf.transpose(r)


            penalty = tf.linalg.band_part((tf.transpose(radii) + radii)/(D + tf.eye(num_rows = D.shape[0])) 
                                           , num_lower= 0, num_upper = 1)



            return tf.reduce_sum(penalty - tf.linalg.tensor_diag(tf.linalg.diag_part(penalty))) 


        prior = tfd.Independent(tfd.Normal(loc = self.m_max_min, 
                    scale =[10.]*self.data_dim), 
            reinterpreted_batch_ndims=1).log_prob(self.upper) +  tfd.Independent(
        tfd.Normal(loc = - self.m_max_min, scale =[10.]*self.data_dim), reinterpreted_batch_ndims=1).log_prob(self.lower)
        #log_p_x_given_k = pik + tf.transpose(
        #New losss:
        return - tf.reduce_sum(list_likelihood) - tf.reduce_sum(prior) #+ noverlap()



    #@tf.function    
    def compute_responsibilities(self, X, y):

        responsibilities =  tf.TensorArray(dtype =tf.float32, size =0, dynamic_size= True)

        # #y_unique = np.unique(y)
        # # tf.map_fn(
        # #     fn, elems, dtype=None, parallel_iterations=None, back_prop=True,
        # #     swap_memory=False, infer_shape=True, name=None
        # # )
        for c in self.y_unique:
            responsibilities = responsibilities.write(c, tf.nn.softmax(self.compute_log_pdf(X, c), axis = 1))

        return responsibilities.stack()#tf.stack([tf.nn.softmax(tf.nn.softmax(self.compute_log_pdf(X, c), axis = 1)) for c in self.y_unique])