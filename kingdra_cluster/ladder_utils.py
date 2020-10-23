
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K


import numpy as np
import tensorflow as tf

import random


class AddBeta(Layer):
    def __init__(self, **kwargs):
        super(AddBeta, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.built:
            return

        self.beta = self.add_weight(name='beta',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
        self.built = True

        super(AddBeta, self).build(input_shape)

    def call(self, x, training=None):
        return tf.add(x,  self.beta)


class G_Guass(Layer):

    def __init__(self, **kwargs):
        super(G_Guass, self).__init__(**kwargs)

    def wi(self,  init, name):
        if init == 1:
            return self.add_weight(name='guess_'+name,
                                   shape=(self.size, ),
                                   initializer='ones',
                                   trainable=True)
        elif init == 0:
            return self.add_weight(name='guess_'+name,
                                   shape=(self.size, ),
                                   initializer='zeros',
                                   trainable=True)

    def build(self, input_shape):

        self.size = input_shape[0][-1]

        self.a1 = self.wi(0., 'a1')
        self.a2 = self.wi(1., 'a2')
        self.a3 = self.wi(0., 'a3')
        self.a4 = self.wi(0., 'a4')
        self.a5 = self.wi(0., 'a5')

        self.a6 = self.wi(0., 'a6')
        self.a7 = self.wi(1., 'a7')
        self.a8 = self.wi(0., 'a8')
        self.a9 = self.wi(0., 'a9')
        self.a10 = self.wi(0., 'a10')

        super(G_Guass, self).build(input_shape)

    def call(self, x):
        z_c, u = x

        a1 = self.a1
        a2 = self.a2
        a3 = self.a3
        a4 = self.a4
        a5 = self.a5
        a6 = self.a6
        a7 = self.a7
        a8 = self.a8
        a9 = self.a9
        a10 = self.a10

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

        z_est = (z_c - mu) * v + mu
        return z_est

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.size)


def add_noise(inputs, noise_std):
    return Lambda(lambda x: x + tf.random_normal(tf.shape(x)) * noise_std)(inputs)


def batch_normalization(batch, mean=None, var=None):
    if mean is None or var is None:
        mean, var = tf.nn.moments(batch, axes=[0])
    return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))
