#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:36:39 2020

@author: gnanfack
"""
import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions

class BlackBoxNN(tf.keras.Model):

    def __init__(self, nb_units, nb_classes, data_dim = 2):
        super(BlackBoxNN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units = nb_units, activation=tf.nn.relu, input_shape= (data_dim,))
        self.dense2 = tf.keras.layers.Dense(nb_units, activation=tf.nn.relu)

        self.classifier = tf.keras.layers.Dense(nb_classes, activation="softmax")




    def share_loss(self, X, sTGMA, weights = None):
        kl = tf.keras.losses.KLDivergence()

        def kl_divergence(x):
            return kl(
                tf.exp(
                    sTGMA.compute_log_conditional_distribution(x)
                    ),
            self.__call__(x),
            sample_weight= weights
            )


        return tfp.monte_carlo.expectation(
            f = kl_divergence,
            samples = X,
            log_prob = sTGMA.log_pdf,
               use_reparametrization= False
        )

    def predict(self, X):
        return tf.nn.softmax(self.__call__(X))

    @tf.function
    def __call__(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)


        return self.classifier(x)

