"""This submodule help user to create metric function,or customize objects or cnn setting"""
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
import tensorflow as tf
import warnings
import numpy
from . import removed_terminal_tensors
from . import Builder
class CategoricalCrossEntropyFactory:
    """This class create and return categorical cross entropy function"""
    def __init__(self, class_number, is_static, weights=None, terminal_signal=None):
        self.class_number = class_number
        self.terminal_signal = terminal_signal
        self.weights = weights
        self.is_static = is_static
    @property
    def cross_entropy(self):
        """return cross entropy function"""
        @rename("loss")
        def static_cross_entropy(y_true, y_pred):
            """calculate static categorical cross entropy between y_true and y_pred"""
            if self.terminal_signal is not None:
                (y_true, y_pred) = removed_terminal_tensors(y_true, y_pred,
                                                            self.class_number,
                                                            self.terminal_signal)
            if self.weights is not None:
                y_true = tf.multiply(y_true, self.weights)
            loss = tf.reduce_mean(categorical_crossentropy(y_true, y_pred))
            return loss
        @rename("loss")
        def dynamic_cross_entropy(y_true, y_pred):
            """calculate dynamic categorical cross entropy between y_true and y_pred"""
            if self.terminal_signal is not None:
                (y_true, y_pred) = removed_terminal_tensors(y_true, y_pred, self.class_number,
                                                            self.terminal_signal)
            weights = tf.divide(1, tf.reduce_sum(y_true, 0))
            inf = tf.constant(numpy.float("inf"))
            where_inf = tf.equal(weights, inf)
            weights = tf.where(where_inf, tf.zeros_like(self.weights), self.weights)
            sum_weights = tf.reduce_sum(self.weights)
            weights = tf.divide(self.weights, sum_weights)
            y_true = tf.multiply(y_true, weights)
            loss = tf.reduce_mean(categorical_crossentropy(y_true, y_pred))
            return loss
        if self.is_static:
            return static_cross_entropy
        else:
            warnings.warn(
                "Dynamic categorical cross entrophy function hasn't complete build yet"
            )
        return dynamic_cross_entropy


class CategoricalAccuracyFactory:
    """This class create and return categorical accuracy function"""
    def __init__(self, class_number, terminal_signal=None):
        self.class_number = class_number
        self.terminal_signal = terminal_signal
    @property
    def accuracy(self):
        """return accuracy function"""
        @rename("accuracy")
        def advanced_categorical_accuracy(y_true, y_pred):
            """calculate categorical accuracy"""
            if self.terminal_signal is not None:
                (y_true, y_pred) = removed_terminal_tensors(y_true, y_pred, self.class_number,
                                                            self.terminal_signal)
            accuracy = tf.reduce_mean(categorical_accuracy(y_true, y_pred))
            return accuracy
        return advanced_categorical_accuracy
def rename(newname):
    """rename the function"""
    def decorator(function):
        """rename input function name"""
        function.__name__ = newname
        return function
    return decorator

class PrecisionFactory:
    """This class create and return precision function"""
    def __init__(self, function_name, number_of_class, target_index, terminal_signal=None):
        self.function_name = function_name
        self.number_of_class = number_of_class
        self.target_index = target_index
        self.terminal_signal = terminal_signal
    @property
    def precision(self):
        """return precision function"""
        @rename(self.function_name)
        def basic_precision(true, pred):
            """calculate the precision"""
            if self.terminal_signal is not None:
                clean_true, clean_pred = removed_terminal_tensors(true, pred,
                                                                  self.number_of_class,
                                                                  self.terminal_signal)
            else:
                clean_true = tf.reshape(true, [-1])
                clean_pred = tf.reshape(pred, [-1])
            numeric_true = tf.cast(tf.equal(tf.argmax(clean_true, 1), self.target_index), tf.int64)
            numeric_pred = tf.cast(tf.equal(tf.argmax(clean_pred, 1), self.target_index), tf.int64)
            true_positive = tf.reduce_sum(tf.multiply(numeric_true, numeric_pred))
            negative = tf.count_nonzero(1-numeric_true)
            true_negative = tf.reduce_sum(tf.multiply(1-numeric_true, 1-numeric_pred))
            false_positive = negative-true_negative
            return  true_positive/(true_positive+false_positive)
        return basic_precision

class RecallFactory:
    """This class create and return recall function"""
    def __init__(self, function_name, number_of_class, target_index, terminal_signal=None):
        self.function_name = function_name
        self.number_of_class = number_of_class
        self.target_index = target_index
        self.terminal_signal = terminal_signal
    @property
    def recall(self):
        """return recall function"""
        @rename(self.function_name)
        def basic_recall(true, pred):
            """calculate the recall"""
            if self.terminal_signal is not None:
                clean_true, clean_pred = removed_terminal_tensors(true, pred,
                                                                  self.number_of_class,
                                                                  self.terminal_signal)
            else:
                clean_true = tf.reshape(true, [-1])
                clean_pred = tf.reshape(pred, [-1])
            numeric_true = tf.cast(tf.equal(tf.argmax(clean_true, 1),
                                            self.target_index),
                                   tf.int64)
            numeric_pred = tf.cast(tf.equal(tf.argmax(clean_pred, 1),
                                            self. target_index),
                                   tf.int64)
            true_positive = tf.reduce_sum(tf.multiply(numeric_true, numeric_pred))
            false_negative = tf.reduce_sum(tf.multiply(numeric_true, 1-numeric_pred))
            return  true_positive/(true_positive+false_negative)
        return basic_recall
class CnnSettingBuilder(Builder):
    """A class which generate setting about multiple convolution layer"""
    def __init__(self):
        super().__init__()
        self.__layers_settings = []
    def clean_layers(self):
        """reset setting"""
        self.__layers_settings = []
        return self
    def add_layer(self, filter_num, filter_size):
        """Add a convolution layer setting"""
        setting = {'filter_num':filter_num, 'filter_size':filter_size}
        self.__layers_settings.append(setting)
        return self
    def _validate(self):
        pass
    def build(self):
        """Get settings"""
        self._validate()
        return self.__layers_settings
