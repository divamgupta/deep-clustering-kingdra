
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K


import numpy as np
import tensorflow as tf

import random

from .unsup_losses import *
from .ladder_utils import *


def get_mi_ladder_model(inp_dim,
                        mi_loss_w_init=0.1,
                        fc_layer_sizes=[1200, 1200, 10],
                        ladder_loss_w_init=1.0,
                        sup_loss_w_init=0,
                        noise_kl_w_init=1,
                        noise_std=0.3,
                        mi_mu_w_init=4, lr=0.002,
                        denoising_cost=[1000.0, 10.0, 0.10, 0.10], last_layer_bn=False):

    mi_loss_w = K.variable(mi_loss_w_init)
    ladder_loss_w = K.variable(ladder_loss_w_init)
    sup_loss_w = K.variable(sup_loss_w_init)
    noise_kl_w = K.variable(noise_kl_w_init)
    mi_mu_w = K.variable(mi_mu_w_init)

    layer_sizes = [inp_dim] + fc_layer_sizes
    L = len(layer_sizes) - 1  # number of layers

    fc_enc = [Dense(s, use_bias=False, kernel_initializer='glorot_normal')
              for s in layer_sizes[1:]]
    fc_dec = [Dense(s, use_bias=False, kernel_initializer='glorot_normal')
              for s in layer_sizes[:-1]]
    betas = [AddBeta() for l in range(L)]

    def encoder(inputs, noise_std):

        h = add_noise(inputs, noise_std)
        all_z = [None for _ in range(len(layer_sizes))]
        all_z[0] = h

        for l in range(1, L+1):

            z_pre = fc_enc[l-1](h)

            if l == L and (not last_layer_bn):
                z = z_pre
            else:
                z = Lambda(batch_normalization)(z_pre)

            z = add_noise(z,  noise_std)

            if l == L:
                h = (betas[l-1](z))
            else:
                h = Activation('relu')(betas[l-1](z))
            all_z[l] = z

        return h, all_z

    inp = Input((inp_dim,))
    inp_sup = Input((inp_dim,))

    corr_logits, corr_z = encoder(inp, noise_std)
    clean_logits,  clean_z = encoder(inp, 0.0)

    corr_logits_sup, _ = encoder(inp_sup, noise_std)
    clean_logits_sup,  _ = encoder(inp_sup, 0.0)

    # Decoder
    d_cost = []
    for l in range(L, -1, -1):
        z, z_c = clean_z[l], corr_z[l]
        if l == L:
            u = corr_logits
        else:
            u = fc_dec[l](z_est)
        u = Lambda(batch_normalization)(u)
        z_est = G_Guass()([z_c, u])
        d_cost.append((tf.reduce_mean(tf.reduce_sum(
            tf.square(z_est - z), 1)) / layer_sizes[l]) * denoising_cost[l])

    ladder_cost = tf.add_n(d_cost)

    corr_logits_sup = Lambda(lambda x: x[0])(
        [corr_logits_sup,  clean_logits, corr_logits, clean_logits_sup, u, z_est, z])
    corr_p_sup = Activation("softmax")(corr_logits_sup)
    clean_p_sup = Activation("softmax")(clean_logits_sup)

    mi_loss_corr = mut_inf_loss(corr_logits, mi_mu_w)
    mi_loss_clean = mut_inf_loss(clean_logits, mi_mu_w)

    noise_kl = tf.reduce_mean(compute_kld(
        tf.stop_gradient(clean_logits), corr_logits))

    tot_loss = ladder_loss_w*ladder_cost + mi_loss_w * \
        (mi_loss_clean + mi_loss_corr) + noise_kl_w*noise_kl

    optimizer = keras.optimizers.Adam(lr=lr)

    te_model = Model(inp_sup, clean_p_sup)
    model = Model([inp, inp_sup], corr_p_sup)
    model.add_loss(tot_loss)
    model.compile(optimizer,  'categorical_crossentropy',
                  loss_weights=[sup_loss_w],  metrics=[])

    try:
        model.metrics_names.append('mi_loss_clean')
        model.metrics_tensors.append(mi_loss_clean)

        model.metrics_names.append('mi_loss_corr')
        model.metrics_tensors.append(mi_loss_corr)

        model.metrics_names.append('ladder_cost')
        model.metrics_tensors.append(ladder_cost)

        model.metrics_names.append('noise_kl')
        model.metrics_tensors.append(noise_kl)
    except Exception as e:
        print(e)

    model.te_model = te_model
    model.mi_loss_w = mi_loss_w
    model.ladder_loss_w = ladder_loss_w
    model.sup_loss_w = sup_loss_w
    model.noise_kl_w = noise_kl_w
    model.mi_mu_w = mi_mu_w

    return model


class LadderIM(object):
    """docstring for LadderIM"""

    def __init__(self, n_epochs_per_iter=3,  n_iter=15,
                 batch_size=64,
                 n_class=10, **model_args):

        self.n_epochs_per_iter = n_epochs_per_iter
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.n_class = n_class
        self.model_args = model_args

        self.model = None

    def fit(self, X_train):

        n_epochs_per_iter = self.n_epochs_per_iter
        n_iter = self.n_iter
        batch_size = self.batch_size
        n_class = self.n_class
        model_args = self.model_args

        model = get_mi_ladder_model(X_train.shape[-1],  **model_args)
        Y_train = np.zeros((X_train.shape[0], n_class))

        for it in range(n_iter):

            model.fit(x=[X_train, X_train], y=Y_train, batch_size=batch_size,
                      epochs=n_epochs_per_iter, shuffle=False)

            Output = model.te_model.predict_on_batch(x=X_train)
            y_pred = np.argmax(Output, axis=1)

        self.model = model

        return y_pred

    def predict(self, X_test):

        Output_test = self.model.te_model.predict_on_batch(x=X_test)
        y_pred_test = np.argmax(Output_test, axis=1)
        return y_pred_test
