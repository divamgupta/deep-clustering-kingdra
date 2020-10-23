

from keras.models import *
import math
import sys
from .unsup_ens_utils import *
from .mi_ladder_unsup import get_mi_ladder_model
import random
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import *
import keras
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


np.random.seed(1)


class KingdraCluster(object):
    """docstring for KingdraCluster"""

    def __init__(self, n_en=10,
                 n_epochs_per_iter0=5,
                 n_epochs_per_iter=5,
                 low_th=1,
                 med_th=4,
                 high_th=9,
                 thresh=0.9,
                 batch_size=64,
                 n_class=10,
                 ens_common_mapping=True,
                 n_iter=40, mi_loss_w_it2=None, **model_args):

        self.n_en = n_en
        self.n_epochs_per_iter0 = n_epochs_per_iter0
        self.n_epochs_per_iter = n_epochs_per_iter
        self.low_th = low_th
        self.med_th = med_th
        self.high_th = high_th
        self.thresh = thresh
        self.batch_size = batch_size
        self.n_class = n_class
        self.n_iter = n_iter
        self.mi_loss_w_it2 = mi_loss_w_it2
        self.model_args = model_args
        self.ens_common_mapping = ens_common_mapping

        self.models = []

    def fit(self, X_train, callback=None, cb2=None):

        n_en = self.n_en
        n_epochs_per_iter0 = self.n_epochs_per_iter0
        n_epochs_per_iter = self.n_epochs_per_iter
        low_th = self.low_th
        med_th = self.med_th
        high_th = self.high_th
        thresh = self.thresh
        batch_size = self.batch_size
        n_class = self.n_class
        n_iter = self.n_iter
        mi_loss_w_it2 = self.mi_loss_w_it2
        model_args = self.model_args
        ens_common_mapping = self.ens_common_mapping

        models = [get_mi_ladder_model(
            X_train.shape[-1],  **model_args) for k in range(n_en)]

        all_outputs = [None for _ in range(n_en)]
        m_inds = [np.zeros((n_class,), dtype=np.int32) for _ in range(n_en)]
        m_buckets = [np.zeros((n_class,), dtype=np.int32) for _ in range(n_en)]

        sys.stdout.flush()
        min_labels = 0
        max_min_labels = 0
        max_labels = 0
        patience = 0
        save_buckets = None
        smooth = False

        self.inp_l = X_train.shape[-1]

        for it in range(n_iter):

            for en in range(n_en):

                model = models[en]

                if it == 0 or min_labels == 0:
                    Y_train_sup = np.zeros((X_train.shape[0], n_class))
                    X_train_sup = X_train
                    model.fit(x=[X_train, X_train_sup], y=Y_train_sup, batch_size=batch_size,
                              epochs=n_epochs_per_iter0, shuffle=False, verbose=2)  # ) callbacks=[lrate]  )
                else:

                    if not ens_common_mapping:
                        m_buckets = get_bucket_modelwrt_ind(
                            buckets, all_outputs[en].argmax(-1))

                    X_train_sup, Y_train_sup = get_training_from_buckets(
                        m_buckets, X_train, it,  smooth)

                    model.fit(x=[X_train, X_train_sup], y=Y_train_sup, batch_size=batch_size,
                              epochs=n_epochs_per_iter, shuffle=True, verbose=2)  # ) callbacks=[lrate]  )

                Output = model.te_model.predict_on_batch(x=X_train)
                y_pred = np.argmax(Output, axis=1)

                if not cb2 is None:
                    cb2(it, y_pred, model)

                all_outputs[en] = Output

                sys.stdout.flush()

            all_outputs1 = np.array(all_outputs)
            buckets = get_en_clus(low_th, med_th, high_th,
                                  all_outputs1, thresh)

            nl = 0
            min_labels = all_outputs1.shape[1]
            for cl in range(n_class):
                count = len(buckets[cl])
                nl += count
                if count < min_labels:
                    min_labels = count

            if max_min_labels < min_labels:
                max_min_labels = min_labels
                save_buckets = buckets[:]
            if min_labels == 0 and save_buckets is not None:
                buckets = save_buckets[:]
                min_labels = max_min_labels

            if min_labels > 0:
                if ens_common_mapping:
                    m_buckets = get_bucket_modelwrt_ind(
                        buckets, all_outputs[en].argmax(-1))

            if it > 0:
                if ens_common_mapping:
                    avg_probs = np.mean(all_outputs1, axis=0)
                    y_pred_e = np.argmax(avg_probs, axis=1)
                else:
                    y_pred_e = np.argmax(all_outputs1[0], axis=1)

                if not callback is None:
                    callback(it, y_pred_e, models)

            for m in models:
                if min_labels > 0:
                    K.set_value(m.sup_loss_w, 1)
                if not mi_loss_w_it2 is None:
                    K.set_value(m.mi_loss_w, mi_loss_w_it2)

        self.models = models

        return y_pred_e

    def save_weights(self, path):
        assert len(self.models) > 0, "The model is not trained yet"

        for i, m in enumerate(self.models):
            m.save_weights(path + "." + str(i))

        np.save(path + "inp_len.npy", self.inp_l)
        np.save(path + "n_en.npy", self.n_en)

    def load_weights(self, path):

        inp_l = int(np.load(path + "inp_len.npy"))
        inp_l = int(np.load(path + "n_en.npy"))

        model_args = self.model_args

        self.models = [get_mi_ladder_model(
            inp_l,  **model_args) for _ in range(n_en)]

        for i, m in enumerate(self.models):
            m.load_weights(path + "." + str(i))

    def predict(self, X):

        assert len(self.models) > 0, "The model is not trained yet"

        all_outputs = []

        for model in self.models:
            Output_test = model.te_model.predict_on_batch(x=X)
            all_outputs.append(Output_test)

        all_outputs1 = np.array(all_outputs)

        if self.ens_common_mapping:
            avg_probs = np.mean(all_outputs1, axis=0)
            y_pred_e = np.argmax(avg_probs, axis=1)
        else:
            y_pred_e = np.argmax(all_outputs1[0], axis=1)

        return y_pred_e
