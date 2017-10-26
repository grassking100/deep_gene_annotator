from . import Input,Dropout,Convolution1D,Flatten,MaxPooling1D,LSTM,Reshape,Activation
from . import keras
from . import Model
from . import tensorflow as tf
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
#input model setting,and return model
def Exon_intron_finder_factory(convolution_settings=[], LSTM_layer_number=1,add_terminal_signal=True,add_batch_normalize=False,dropout=0,learning_rate=0.001,output_dim=1):
    #initialize the variable#####
    seq_size=None
    code_dim=4
    seq_input_shape=(seq_size,code_dim)
    input_layers=Input(shape=seq_input_shape, name='Input')
    previous_layer=input_layers
    index=1
    ############################
    #generate every convolution layer with input's setting
    for setting in convolution_settings:
        bias=not add_batch_normalize
        convolution=Convolution1D(filters=setting['filter_num'],
                                  kernel_size=setting['filter_size'], padding='same',
                                  name='Convolution'+(str)(index),
                                  input_shape=(seq_size,code_dim),
                                  use_bias=bias)(previous_layer)
        previous_layer=convolution
        if add_batch_normalize:
            batch=keras.layers.normalization.BatchNormalization(name='BatchNormal'+(str)(index))(previous_layer)
            previous_layer=batch
        previous_layer=Activation('relu')(previous_layer)
        index+=1
    #choose algorithm to calculate the loss and accuracy
    if add_terminal_signal:
        loss_func=tensor_end_with_terminal_binary_crossentropy
        loss_acc=tensor_end_with_terminal_binary_accuracy
    else:
        loss_func=keras.losses.binary_crossentropy
        loss_acc=keras.metrics.binary_accuracy
    #add lstm rnn layer#  
    lstm_forward=LSTM(LSTM_layer_number,return_sequences=True ,activation='tanh',name='Lstm_forward',dropout=dropout)(previous_layer)
    lstm_backword=LSTM(LSTM_layer_number,return_sequences=True ,go_backwards=True,activation='tanh',name='Lstm_backword',dropout=dropout)(previous_layer)
    ####################
    #concatenate two lstm rnn layers
    concatenate = keras.layers.concatenate(inputs=[lstm_forward,lstm_backword],name='Concatenate')
    #create last layer to predict the result
    if output_dim==1:
        last_activation='sigmoid'
    else:
        last_activation='softmax'
    last=Convolution1D(activation=last_activation,filters=output_dim, kernel_size=1,name='Result')(concatenate)
    #create model
    Exon_intron_finder= Model(inputs=input_layers, outputs=last)
    #set optimizer metrics,and loss to the model
    keras.optimizers.Adam(lr=learning_rate)
    Exon_intron_finder.compile(optimizer='adam', loss=loss_func, metrics=[loss_acc])
    return Exon_intron_finder

    