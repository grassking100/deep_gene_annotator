#get all index which corresponding value equal -1 
def tensor_non_terminal_index(tensor):
    index=tf.where(tf.not_equal(tensor,-1))
    return tf.reshape(index, [-1])
#calculate binary crossentropy between y_true and y_pred which have terminal signal -1
def tensor_end_with_terminal_binary_crossentropy(y_true,y_pred):
    y_true=tf.reshape(y_true, [-1])
    y_pred=tf.reshape(y_pred, [-1])
    index=tensor_non_terminal_index(y_true)
    loss=keras.losses.binary_crossentropy(
        tf.gather(tf.cast(y_true,tf.float32),index),
        tf.gather(y_pred,index))
    return loss
#calculate binary accuracy between y_true and y_pred which have terminal signal -1
def tensor_end_with_terminal_binary_accuracy(y_true,y_pred):
    y_true=tf.reshape(y_true, [-1])
    y_pred=tf.reshape(y_pred, [-1])
    index=tensor_non_terminal_index(y_true)
    accuracy=keras.metrics.binary_accuracy(
        tf.gather(tf.cast(y_true,tf.float32),index),
        tf.gather(y_pred,index))
    return accuracy
#A class which generate setting about multiple convolution layer
class Convolution_layers_settings():
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