from . import Input,Dropout,Convolution1D,Flatten,MaxPooling1D,LSTM,Reshape,Activation
from . import keras
from . import Model
from . import tensor_end_with_terminal_binary_accuracy,tensor_end_with_terminal_binary_crossentropy
#input model setting,and return model
def SeqAnnModelFactory(convolution_settings=[], LSTM_layer_number=1,add_terminal_signal=True,add_batch_normalize=False,dropout=0,learning_rate=0.001,output_dim=1):
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

    