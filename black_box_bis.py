#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:36:39 2020

@author: gnanfack
"""
import tensorflow as tf

import tensorflow_probability as tfp

tfd = tfp.distributions

def persoBlackBoxNN( nb_classes, data_dim, dataset_name, nb_units = [128, 128]):

    switcher={
        "magic_gamma": BlackBoxMagicGamma,
        "ionosphere": BlackBoxBankMarketing
    }

    return switcher[dataset_name](nb_units = nb_units, nb_classes = nb_classes, data_dim = data_dim)



class BlackBoxBankMarketing(tf.Module):

    def __init__(self, nb_units, nb_classes, data_dim = 2):
        super(BlackBoxBankMarketing, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units = nb_units[0], activation=tf.nn.elu, input_shape= (data_dim,), kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.dense3 = tf.keras.layers.Dense(256, activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.classifier = tf.keras.layers.Dense(nb_classes, activation="softmax")

    @tf.function
    def losses(self):
        return self.dense1.losses + self.dense2.losses + self.dense3.losses #+ self.classifier.losses

    @tf.function
    def share_loss(self, X, sTGMA, weights = None):
        kl = tf.keras.losses.KLDivergence()
        print("----tracing-shareloss")
        @tf.function
        def kl_divergence(x):
            return kl(
                tf.exp(
                    sTGMA.compute_log_conditional_distribution(x)
                    ),
            self.__call__(x, trainable=tf.constant(True)),
            sample_weight= weights
            )


        return tfp.monte_carlo.expectation(
            f = kl_divergence,
            samples = X,
            log_prob = sTGMA.log_pdf,
               use_reparametrization= False
        )

    @tf.function
    def predict(self, X, trainable):
        return tf.nn.softmax(self.__call__(X, trainable))

    @tf.function
    def __call__(self, inputs, trainable ):
        x = self.dense1(inputs)
        x = self.dropout1(x, training= trainable)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout2(x, training= trainable)



        return self.classifier(x)


class BlackBoxMagicGamma(tf.Module):

    def __init__(self, nb_units, nb_classes, data_dim = 2):
        super(BlackBoxMagicGamma, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units = 40, activation=tf.nn.elu, input_shape= (data_dim,), kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.dense2 = tf.keras.layers.Dense(25, activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.dense3 = tf.keras.layers.Dense(10, activation=tf.nn.elu, kernel_regularizer=tf.keras.regularizers.l2(0.00001))
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.classifier = tf.keras.layers.Dense(nb_classes, activation="softmax")

    @tf.function
    def losses(self):
        return self.dense1.losses + self.dense2.losses + self.dense3.losses #+ self.classifier.losses

    @tf.function
    def share_loss(self, X, sTGMA, weights = None):
        kl = tf.keras.losses.KLDivergence()
        print("----tracing-shareloss")
        @tf.function
        def kl_divergence(x):
            return kl(
                tf.exp(
                    sTGMA.compute_log_conditional_distribution(x)
                    ),
            self.__call__(x, trainable=tf.constant(True)),
            sample_weight= weights
            )


        return tfp.monte_carlo.expectation(
            f = kl_divergence,
            samples = X,
            log_prob = sTGMA.log_pdf,
               use_reparametrization= False
        )

    @tf.function
    def predict(self, X, trainable):
        return tf.nn.softmax(self.__call__(X, trainable))

    @tf.function
    def __call__(self, inputs, trainable ):
        x = self.dense1(inputs)
        x = self.dropout1(x, training= trainable)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dropout2(x, training= trainable)



        return self.classifier(x)
