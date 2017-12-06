from . import tensorflow as tf
from . import keras
from . import warnings
def removed_terminal_tensors(true,pred,number_of_class,value_to_ignore):
    reshape_pred=tf.reshape(pred,[-1])
    reshape_true=tf.reshape(true,[-1])
    index=tf.where(tf.not_equal(reshape_true,[value_to_ignore]))
    clean_pred=tf.reshape(tf.gather(reshape_pred,index),[-1,number_of_class])
    clean_true=tf.reshape(tf.gather(reshape_true,index),[-1,number_of_class])
    return (clean_true,clean_pred)
def tensor_non_terminal_index(tensor):
    warnings.warn(
            "tensor_non_terminal_index is deprecated,it will be removed in the future",
            PendingDeprecationWarning
    )
    #get all index which corresponding value equal -1 
    index=tf.where(tf.not_equal(tensor,-1))
    return tf.reshape(index, [-1])
def categorical_crossentropy_factory(class_number,weights=None,terminal_signal=None):
    def categorical_crossentropy(y_true,y_pred):
        #calculate categorical crossentropy between y_true and y_pred
        if terminal_signal is not None:
            (y_true,y_pred)=removed_terminal_tensors(y_true,y_pred,class_number,terminal_signal)
        if weights is not None:
            loss=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred,tf.constant(weights)))
        else:            
            loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred))
        return loss
    return categorical_crossentropy
def categorical_accuracy_factory(class_number,terminal_signal=None):                            
    def categorical_accuracy(y_true,y_pred):
        #calculate categorical crossentropy between y_true and y_pred
        if terminal_signal is not None:                         
            (y_true,y_pred)=removed_terminal_tensors(y_true,y_pred,class_number,terminal_signal)
        accuracy=tf.reduce_mean(keras.metrics.categorical_accuracy(y_true,y_pred))
        return accuracy
    return categorical_accuracy
def tensor_end_with_terminal_binary_crossentropy(y_true,y_pred):
    #calculate binary crossentropy between y_true and y_pred which have terminal signal -1
    warnings.warn(
            "tensor_end_with_terminal_binary_crossentropy is deprecated,it will be removed in the future",
            PendingDeprecationWarning
    )
    y_true=tf.reshape(y_true, [-1])
    y_pred=tf.reshape(y_pred, [-1])
    index=tensor_non_terminal_index(y_true)
    loss=keras.losses.binary_crossentropy(
        tf.gather(tf.cast(y_true,tf.float32),index),
        tf.gather(y_pred,index))
    return loss
def tensor_end_with_terminal_binary_accuracy(y_true,y_pred):
    #calculate binary accuracy between y_true and y_pred which have terminal signal -1
    warnings.warn(
            "tensor_end_with_terminal_binary_accuracy is deprecated,it will be removed in the future",
            PendingDeprecationWarning
    )
    y_true=tf.reshape(y_true, [-1])
    y_pred=tf.reshape(y_pred, [-1])
    index=tensor_non_terminal_index(y_true)
    accuracy=keras.metrics.binary_accuracy(
        tf.gather(tf.cast(y_true,tf.float32),index),
        tf.gather(y_pred,index))
    return accuracy

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

def precision_creator(function_name,number_of_class,target_index,terminal_signal=None):
    @rename(function_name)
    def precision(true,pred):
        if terminal_signal is not None:
            clean_true,clean_pred=removed_terminal_tensors(true,pred,number_of_class,terminal_signal)
        else:
            clean_true=tf.reshape(true,[-1])
            clean_pred=tf.reshape(pred,[-1])
        numeric_true=tf.cast(tf.equal(tf.argmax(clean_true, 1),target_index),tf.int64)
        numeric_pred=tf.cast(tf.equal(tf.argmax(clean_pred, 1),target_index),tf.int64)
        TP=tf.reduce_sum(tf.multiply(numeric_true,numeric_pred))
        N=tf.count_nonzero(1-numeric_true)
        TN=tf.reduce_sum(tf.multiply(1-numeric_true,1-numeric_pred))
        FP=N-TN
        return  TP/(TP+FP)
        #return None
    return precision

def recall_creator(function_name,number_of_class,target_index,terminal_signal=None):
    @rename(function_name)
    def recall(true,pred):
        if terminal_signal is not None:
            clean_true,clean_pred=removed_terminal_tensors(true,pred,number_of_class,terminal_signal)
        else:
            clean_true=tf.reshape(true,[-1])
            clean_pred=tf.reshape(pred,[-1])
        numeric_true=tf.cast(tf.equal(tf.argmax(clean_true, 1),target_index),tf.int64)
        numeric_pred=tf.cast(tf.equal(tf.argmax(clean_pred, 1),target_index),tf.int64)
        TP=tf.reduce_sum(tf.multiply(numeric_true,numeric_pred))
        FN=tf.reduce_sum(tf.multiply(numeric_true,1-numeric_pred))
        return  TP/(TP+FN)
    return recall

class CnnSettingCreator():
    #A class which generate setting about multiple convolution layer
    def __init__(self):
        self.layers_settings=[]
    def clean_layers(self):
        self.layers_settings=[]
        return self
    def add_layer(self,filter_num,filter_size):
        setting={'filter_num':filter_num,'filter_size':filter_size}
        self.layers_settings.append(setting)
        return self
    def get_settings(self):
        return self.layers_settings