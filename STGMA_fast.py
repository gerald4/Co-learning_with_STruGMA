import os
import numpy as np
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
from truncated_normal import SoftTruncatedNormal

from config import  config_params, hyper_params
tfd = tfp.distributions




class SoftTruncatedGaussianMixtureAnalysis(tf.Module):

    def __init__(self, n_components, data_dim, n_classes, theta, seed = 111, m_max_min = 10.):
        """
        This function creates the variables of the model.
        n_components: number of components for each mixture per class
        data_dim: number of features
        n_classes: number of classes
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)

        super(SoftTruncatedGaussianMixtureAnalysis, self).__init__()

        #Value eta for logisitic
        self.eta = tf.Variable(20.0, trainable=False)

        self.n_components = n_components
        self.stable = tf.constant(np.finfo(np.float32).eps)

        self.data_dim = tf.constant(data_dim)
        self.n_classes = tf.constant(n_classes)

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
        self.y_unique = tf.range(n_classes)


    def gmm_initialisation(self, X_train, y_train):
        """This function intialises our STGMA using gaussian mixture model
        """

        #Split X_train by unique Y_train
        y_unique = np.unique(y_train)
        #For each unique Y_train, create a GMM and initialise parameters of our STGMA
        for i in range(len(y_unique)):
            gmm = GaussianMixture(n_components = self.n_components,
                               covariance_type="diag", reg_covar=1e-02)
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

    @tf.function    
    def normalizing_constant(self, way = "independent"):
        """This function computes the normalizing constant which envolves the integral
        """
        d = tfd.Normal(loc = self.mu, scale = np.finfo(np.float32).eps + tf.nn.softplus(self.sigma))
        return tf.reduce_prod(d.cdf(self.upper) - d.cdf(self.lower), axis = -1, keepdims=True) + self.stable

    @tf.function
    def compute_log_pdf(self, X, y):
        dist = tfd.Mixture(
          cat = tfd.Categorical(logits = self.logits_k[y]),
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
    # @tf.function#(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),                                   tf.TensorSpec(shape=[], dtype=tf.int32)))
    # def compute_log_pdf(self, X, y):
    #     print("---Tracing--- compute_log_pdf")

    #     #def compute_log_pdf_single(ind):
    #     return tfd.Independent(SoftTruncatedNormal(loc = self.mu[y],
    #                                           scale = np.finfo(np.float32).eps + tf.nn.softplus(self.sigma[y]),
    #                                         low = self.lower[y],
    #                                         high = self.upper[y],
    #                                         eta = self.eta),
    #                           reinterpreted_batch_ndims=1).log_prob(tf.expand_dims(X,1)) + tf.expand_dims(self.logits_k[y],0)
                              # + self.logits_k[ind][]
        #return  tf.map_fn(compute_log_pdf_single, tf.range(tf.shape(X)[0]), fn_output_signature=tf.float32, parallel_iterations = 1000)


    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                   tf.TensorSpec(shape=[None], dtype=tf.int32),
                                   tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)))
    def projection(self, X, y, resp, weights, t_range):

        print("----Tracing___projection")

        @tf.function
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


        tmp_indexes = tf.where(tf.less(self.no_ovelap_test(), -self.theta/50.))
        #tf.print("number of remaining overlapping: ", tf.size(tmp_indexes))


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



    @tf.function
    def no_ovelap_test(self):
        low = tf.reshape(self.lower, (self.lower.shape[0]*self.lower.shape[1],)+ (self.lower.shape[2:]))
        
        upp = tf.reshape(self.upper, (self.upper.shape[0]*self.upper.shape[1],)+ (self.upper.shape[2:]))
        
        centers = (1./2)*(upp + low)
        
        lenghts = (1./2)*(upp - low)
        
        pairwise_centers = tf.abs(centers[tf.newaxis, ...] - centers[:, tf.newaxis,...])
        
        pairwise_lengths = lenghts[tf.newaxis, ...] + lenghts[:, tf.newaxis, ...]
        
        no_overlap_mat = tf.reduce_max(pairwise_centers - pairwise_lengths, axis=-1)

        return tf.linalg.band_part(no_overlap_mat, -1, 0) - tf.linalg.band_part(no_overlap_mat, 0, 0)



    @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.float32),))
    def sample_importance(self, nb_samples):
        i0 = tf.constant(0)
        samp = tf.reshape(tf.convert_to_tensor(()), (0, self.data_dim))
        cond = lambda y, samp: y < self.y_unique.shape[0]
        
        @tf.function
        def funct(y, samp):
            
            n_samples = tf.dtypes.cast(self.logits_y[y]*nb_samples, tf.int32)
            dist = tfd.Mixture( cat = tfp.distributions.Categorical(
                 logits = self.logits_k[y]),
                 components = [
                     tfd.Independent(
                         SoftTruncatedNormal(loc = self.mu[y,j,:],
                            low = self.lower[y,j,:],
                            high = self.upper[y,j,:],
                            scale = np.finfo(np.float32).eps + tf.nn.softplus(
                                self.sigma[y,j,:]),
                            eta = self.eta
                            ),
                         reinterpreted_batch_ndims=1) for j in range(
                             self.n_components
                             )
                             ]
                             )
            return [y+1, tf.concat([samp, dist.sample(n_samples)], axis = 0)]
                    

        return tf.while_loop(
            cond, funct, loop_vars=[i0, samp],
            shape_invariants=[i0.get_shape(), tf.TensorShape([None, self.data_dim])])[1]

    @tf.function#(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def log_joint_prob(self, X):
        print("----Tracing-log_joint_prob")
        before_reduce_sum = tf.map_fn(lambda y: self.compute_log_pdf(X,y), self.y_unique, fn_output_signature=tf.float32)
        
        reduce_sum =  tf.reduce_logsumexp(before_reduce_sum, axis = -1) + tf.expand_dims(tf.math.log(self.logits_y + self.stable), axis =1)
        return tf.transpose(reduce_sum)
        
    @tf.function#(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def log_pdf(self, X):
        return tf.reduce_logsumexp(self.log_joint_prob(X), axis = -1)

    @tf.function#(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
    def compute_log_conditional_distribution(self, X):
        print("---Tracing---log_conditional_distribub")
        before_reduce_sum = tf.map_fn(lambda y: self.compute_log_pdf(X,y), self.y_unique, fn_output_signature=tf.float32)
        
        reduce_sum =  tf.reduce_logsumexp(before_reduce_sum, axis = -1) + tf.expand_dims(tf.math.log(self.logits_y + self.stable), axis =1)

        log_joint_prob = tf.transpose ( reduce_sum)

        return log_joint_prob - tf.reduce_logsumexp(log_joint_prob, axis = -1, keepdims = True)
    
    @tf.function
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

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),))
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

    @tf.function
    def expected_ll (self, X, y, responsibilities, weights):

#         list_likelihood = tf.TensorArray(dtype =tf.float32, size =0, dynamic_size= True)
#         #For each unique y_train, create a compute the expected log-likelihood
        
        @tf.function
        def multiply_t(y):
            return  tf.reduce_mean(
                    tf.multiply(
                        tf.multiply(weights[:,y, tf.newaxis],
                                responsibilities[y]
                            ),
                            self.compute_log_pdf(
                            X, 
                            y

                            )
                    )
            )
                        
        before_reduce_mean = tf.map_fn(lambda y: multiply_t(y), self.y_unique, fn_output_signature=tf.float32)

        return - tf.reduce_sum(before_reduce_mean)


    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                   tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, None], dtype=tf.float32)))
    def __call__(self, X, y, responsabilities, weights):

        #Compute expected likelihood per class
        print("----Tracing---call")
        @tf.function
        def multiply_tt(y):
            return  tf.reduce_mean(
                    tf.multiply(
                        tf.multiply(weights[:,y, tf.newaxis],
                                responsabilities[y]
                            ),
                            self.compute_log_pdf(
                            X, 
                            y

                            )
                    )
            )
                        
        before_reduce_mean = tf.map_fn(lambda y: multiply_tt(y), self.y_unique, fn_output_signature=tf.float32)
        prior = tfd.Independent(tfd.Normal(loc = self.m_max_min, 
                    scale =10.*tf.ones(self.data_dim)), 
            reinterpreted_batch_ndims=1).log_prob(self.upper) +  tfd.Independent(
        tfd.Normal(loc = - self.m_max_min, scale =10.*tf.ones(self.data_dim)), reinterpreted_batch_ndims=1).log_prob(self.lower)
        #log_p_x_given_k = pik + tf.transpose(
        #New losss:
        return - tf.reduce_sum(before_reduce_mean) - tf.reduce_sum(prior)

    @tf.function(input_signature=(tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                                   tf.TensorSpec(shape=[None], dtype=tf.int32)
                                 ))   
    def compute_responsibilities(self, X, y):

        #responsibilities =  tf.TensorArray(dtype =tf.float32, size =0, dynamic_size= True)

        # #y_unique = np.unique(y)
        return tf.map_fn(
            lambda y:tf.nn.softmax(self.compute_log_pdf(X,y), axis = 1), self.y_unique, dtype=tf.float32
        )