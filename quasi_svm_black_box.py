import tensorflow as tf

import tensorflow_probability as tfp
from loc_func_tensorflow.kernelized import RandomFourierFeatures


tfd = tfp.distributions

class quasiSVM(tf.Module):

    def __init__(self, nb_units, nb_classes, data_dim = 2):
        super(quasiSVM, self).__init__()
        super(quasiSVM, self).__init__()
        self.features = RandomFourierFeatures(output_dim = nb_units[0], kernel_initializer="gaussian", trainable=True)


        self.logits = tf.keras.layers.Dense(nb_classes,  kernel_regularizer=tf.keras.regularizers.l2(0.1))
    
    @tf.function
    def __call__(self, inputs):
        x = self.features(inputs)


        return self.logits(x)

    @tf.function
    def losses(self):
        return self.logits.losses
    
    @tf.function
    def share_loss(self, X, sTGMA, weights = None):
        kl = tf.keras.losses.KLDivergence()

        def kl_divergence(x):
            return kl(
                tf.exp(
                    sTGMA.compute_log_conditional_distribution(x)
                    ),
            self.predict(x),
            sample_weight= weights
            )


        return tfp.monte_carlo.expectation(
            f = kl_divergence,
            samples = X,
            log_prob = sTGMA.log_pdf,
               use_reparametrization= False
        )

    @tf.function
    def predict(self, X):
        return tf.nn.softmax(self.__call__(X))


