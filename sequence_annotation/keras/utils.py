from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
import tensorflow as tf
from ..utils.exception import LengthNotEqualException

def process_tensor(answer, prediction,values_to_ignore=None):
    """Remove specific ignored singal and concatenate them into a two dimension tensors"""
    true_shape = K.int_shape(answer)
    pred_shape = K.int_shape(prediction)
    if len(true_shape)!=3 or len(pred_shape)!=3:
        raise Exception("Shape must be 3 dimensions")
    if values_to_ignore is not None:
        if not isinstance(values_to_ignore,list):
            values_to_ignore=[values_to_ignore]
        index = tf.where(K.any(K.not_equal(answer,values_to_ignore),-1))
        clean_answer = tf.gather_nd(answer, index)
        clean_prediction = tf.gather_nd(prediction, index)
    else:
        true_label_number = K.shape(answer)[-1]
        pred_label_number = K.shape(prediction)[-1]
        clean_answer = K.reshape(answer, [-1, true_label_number])
        clean_prediction = K.reshape(prediction, [-1,pred_label_number])
    true_shape = K.int_shape(clean_answer)
    pred_shape = K.int_shape(clean_prediction)
    if true_shape != pred_shape:
        raise LengthNotEqualException(true_shape, pred_shape)
    return (clean_answer, clean_prediction)

def padding(inputs, answers, padding_signal):
    align_inputs = pad_sequences(inputs, padding='post',value=0)
    align_answers = pad_sequences(answers, padding='post',value=padding_signal)
    return (align_inputs, align_answers)

def model_method(model,input_index,output_index):
    return K.function([model.layers[input_index].input], [model.layers[output_index].output])