from . import Input,Dropout,Convolution1D,Flatten,MaxPooling1D,LSTM,Reshape,Activation
from . import keras
from . import Model
from . import tensor_end_with_terminal_binary_accuracy,tensor_end_with_terminal_binary_crossentropy
from . import tensor_end_with_terminal_categorical_crossentropy,tensor_end_with_terminal_categorical_accuracy
from . import precision_creator,recall_creator,rename
from . import tensorflow as tf
from . import SeqAnnModel
def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.
    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keep_dims` is `True`,
            the reduced dimensions are retained with length 1.
    # Returns
        A tensor with the mean of elements of `x`.
    """
    if x.dtype.base_dtype == tf.bool:
        x = tf.cast(x, floatx())
    x=[x]
    x=tf.boolean_mask(x, tf.logical_not(tf.is_nan(x)))
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)    
keras.backend.mean=mean




def SeqAnnModelFactory(convolution_settings=[], LSTM_layer_number=1,add_terminal_signal=True,add_batch_normalize=False,dropout=0,learning_rate=0.001,output_dim=1):
    #input model setting,and return model
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
        loss_func=tensor_end_with_terminal_categorical_crossentropy
        loss_acc=tensor_end_with_terminal_categorical_accuracy
        terminal_signal=-1
    else:
        loss_func=keras.losses.categorical_crossentropy
        loss_acc=keras.metrics.categorical_accuracy
        terminal_signal=None
    precisions=[precision_creator("precision_"+str(i),output_dim,i,terminal_signal) for i in range(0,output_dim)]
    recalls=[recall_creator("recall_"+str(i),output_dim,i,terminal_signal) for i in range(0,output_dim)]
    #add lstm rnn layer#  
    lstm_forward=LSTM(LSTM_layer_number,return_sequences=True ,activation='tanh',name='LSTM_Forward',dropout=dropout)(previous_layer)
    lstm_backword=LSTM(LSTM_layer_number,return_sequences=True ,go_backwards=True,activation='tanh',name='LSTM_Backword',dropout=dropout)(previous_layer)
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
    Exon_intron_finder= SeqAnnModel(inputs=input_layers, outputs=last)
    #set optimizer metrics,and loss to the model
    keras.optimizers.Adam(lr=learning_rate)
    loss_acc.__name__="accuracy"
    Exon_intron_finder.compile(optimizer='adam', loss=loss_func, metrics=precisions+recalls+[loss_acc])
    return Exon_intron_finder

    