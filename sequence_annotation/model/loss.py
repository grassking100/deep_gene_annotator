"""This submodule help user to create loss function,or accuracy function for sequence data"""
import warnings
import tensorflow as tf
import keras.backend as K
import numpy as np
from ..data_handler.data_handler import SeqAnnDataHandler
from ..utils.python_decorator import rename
from .metric import SeqAnnMetric

def focal_Loss(y_true, y_pred, alpha=0.25, gamma=2):
    y_pred += K.epsilon()
    ce = -y_true * K.log(y_pred)
    weight_ = np.power(1 - y_pred, gamma) * y_true
    fl = ce * weight_ * alpha
    reduce_fl = K.max(fl, axis=-1)
    return reduce_fl

class Loss(SeqAnnMetric):
    """This class create and return loss function"""
    def __init__(self, name=None, weights=None, values_to_ignore=None,
                 type_='categorical_crossentropy', dynamic_weight_method=None):
        super().__init__(name or type_,values_to_ignore)
        self._weights = weights
        self._dynamic_weight_method = dynamic_weight_method
        self._type = type_
        if type_ == 'focal_loss':
            self._loss_function = focal_Loss
        else:
            try:
                exec('from keras.losses import {type_}'.format(type_=type_))
                exec('self._loss_function={type_}'.format(type_=type_))
            except ImportError as e:
                raise Exception(type_+' cannot be found in keras.losses')
    def get_result(self):
        true,pred = self._true,self._pred
        weight_ = None
        if self._weights is not None:
            weight_ = list(self._weights)
        else:
            if self._dynamic_weight_method is not None:
                warnings.warn("Weight will be recalucalated by dynamic weight")
                if self._dynamic_weight_method=="reversed_count_weight":
                    weight_ = self._reversed_count_weight(true)
                else:
                    raise Exception(self._dynamic_weight_method+" has not be implemented yet")
        if weight_ is not None:
            if self._type=='categorical_crossentropy':
                true = tf.multiply(true, weight_)
            else:
                raise Exception(self._type+" doesn't support weights loss yet")
        loss = tf.reduce_mean(self._loss_function(true,pred))
        return loss
    def _reversed_count_weight(self,seq_tensor):
        dim = tf.cast(tf.shape(seq_tensor)[1], tf.float32)
        class_count = tf.cast(tf.reduce_sum(seq_tensor,[0]), tf.float32)
        reversed_count = tf.divide(1,(class_count+1))
        reversed_count_sum = tf.reduce_sum(reversed_count)
        weight = tf.divide(tf.multiply(1.0,reversed_count),reversed_count_sum)
        return tf.multiply(weight,dim)
    def __call__(self, y_true, y_pred):
        self.set_data(y_true, y_pred)
        return self.get_result()