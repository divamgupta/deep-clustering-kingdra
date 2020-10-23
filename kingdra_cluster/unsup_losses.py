
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K
import numpy as np
import tensorflow as tf


def entropy(p):
    return -tf.reduce_sum(p * tf.log(p + 1e-16), axis=1)


def compute_kld(p_logit, q_logit):
    p = tf.nn.softmax(p_logit)
    q = tf.nn.softmax(q_logit)
    return tf.reduce_sum(p*(tf.log(p + 1e-16) - tf.log(q + 1e-16)), axis=1)


def mut_inf_loss(p_logit, mu):
    p = tf.nn.softmax(p_logit)
    p_ave = tf.reduce_mean(p, axis=0)
    loss_eq2 = -tf.reduce_sum(p_ave * tf.log(p_ave + 1e-16))
    loss_eq1 = tf.reduce_mean(entropy(p))
    loss_eq = loss_eq1 - mu * loss_eq2
    return loss_eq


def normalize_l2(x):
    y = K.pow(K.sum(x**2, axis=1), 0.5)
    y = K.expand_dims(y,  1)
    y = K.repeat_elements(y, x.shape[1], axis=1)
    return x/y


def self_dot_loss(p_logit, l='mse'):
    p = tf.nn.softmax(p_logit)
    p = normalize_l2(p)
    d = K.dot(p, K.transpose(p))
    print("d.shape",  d.shape)
    y = tf.eye(K.shape(d)[0])

    if l == 'bce':
        return K.mean(keras.metrics.binary_crossentropy(y, d))
    elif l == 'mse':
        return K.mean(keras.metrics.mean_squared_error(y, d))


def eye_dot_loss(p_logit, p_logit2, l='mse'):
    p = tf.nn.softmax(p_logit)
    p = normalize_l2(p)

    p2 = tf.nn.softmax(p_logit2)
    p2 = normalize_l2(p2)

    d = K.dot(p, K.transpose(p2))
    print("d.shape",  d.shape)
    y = tf.eye(K.shape(d)[0])

    if l == 'bce':
        return K.mean(keras.metrics.binary_crossentropy(y, d))
    elif l == 'mse':
        return K.mean(keras.metrics.mean_squared_error(y, d))


def random_noise_loss(x, ul_logits):

    d = tf.random_normal(shape=tf.shape(x))
    d /= (tf.reshape(tf.sqrt(tf.reduce_sum(tf.pow(d, 2.0), axis=1)),
                     [-1, 1]) + 1e-16)
    ul_logits = tf.stop_gradient(ul_logits)
    y1 = ul_logits
    y2 = enc(x + d)
    return tf.reduce_mean(compute_kld(y1, y2))


def self_dot_loss_augment(p_logit, l='mse'):
    p = tf.nn.softmax(p_logit)
    p = normalize_l2(p)
    d = K.dot(p, K.transpose(p))
    print("d.shape",  d.shape)
    y = tf.eye(K.shape(d)[0])

    if l == 'bce':
        return K.mean(keras.metrics.binary_crossentropy(y, d))
    elif l == 'mse':
        return K.mean(keras.metrics.mean_squared_error(y, d))


def zero(a, b):
    return K.mean(a + b)*0


def virtual_randnoise_loss(x, ul_logits, enc, rnd_std_dev=0.5):
    r_vadv = rnd_std_dev*tf.random_normal(shape=tf.shape(x))
    ul_logits = tf.stop_gradient(ul_logits)
    y1 = ul_logits
    y2 = enc(x + r_vadv)
    return tf.reduce_mean(compute_kld(y1, y2))


def get_randnoise_loss(inp, logits, enc, rnd_std_dev=0.5):
    return virtual_randnoise_loss(inp, logits, enc, rnd_std_dev=rnd_std_dev)
