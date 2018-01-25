"""This submodule contains class which can create model"""
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.optimizers import Adam
from . import Input, Convolution1D, LSTM, Activation
from . import SeqAnnModel
from . import Builder
class SeqAnnModelBuilder(Builder):
    """This class will create and return sequence annotation model"""
    def __init__(self):
        self.__cnn_setting = None
        self.__lstm_layer_number = None
        self.__add_batch_normalize = None
        self.__learning_rate = None
        self.__output_dim = None
        self.__dropout = None
        self.__custom_objects = None
    @property
    def cnn_setting(self):
        """Get convolution layer setting"""
        return self.__cnn_setting
    @property
    def lstm_layer_number(self):
        """Get lstm layer number"""
        return self.__lstm_layer_number
    @property
    def add_batch_normalize(self):
        """Get boolean about decide to use batch normalize or not"""
        return self.__add_batch_normalize
    @property
    def dropout(self):
        """Get dropout"""
        return self.__dropout
    @property
    def learning_rate(self):
        """Get learning rate"""
        return self.__learning_rate
    @property
    def output_dim(self):
        """Get output dimention size"""
        return self.__output_dim
    @property
    def custom_objects(self):
        """Get custom objects"""
        return self.__custom_objects
    @cnn_setting.setter
    def cnn_setting(self, cnn_setting):
        """Set convolution layer setting"""
        self.__cnn_setting = cnn_setting
        return self
    @lstm_layer_number.setter
    def lstm_layer_number(self, lstm_layer_number):
        """Set lstm layer number"""
        self.__lstm_layer_number = lstm_layer_number
        return self
    @add_batch_normalize.setter
    def add_batch_normalize(self, add_batch_normalize):
        """Decide to use batch normalize or not"""
        self.__add_batch_normalize = add_batch_normalize
        return self
    @dropout.setter
    def dropout(self, dropout):
        """Set dropout"""
        self.__dropout = dropout
        return self
    @learning_rate.setter
    def learning_rate(self, learning_rate):
        """Set learning rate"""
        self.__learning_rate = learning_rate
        return self
    @output_dim.setter
    def output_dim(self, output_dim):
        """Set output dimention size"""
        self.__output_dim = output_dim
        return self
    @custom_objects.setter
    def custom_objects(self, custom_objects):
        """Set custom_objects"""
        self.__custom_objects = custom_objects
        return self
    def build(self):
        """Create and return model"""
        self._validate()
        seq_size = None
        code_dim = 4
        seq_input_shape = (seq_size, code_dim)
        input_layers = Input(shape=seq_input_shape, name='Input')
        previous_layer = input_layers
        ############################
        #generate every convolution layer with input's setting
        for index, setting in enumerate(self.__cnn_setting):
            bias = not self.__add_batch_normalize
            convolution = Convolution1D(filters=setting['filter_num'],
                                        kernel_size=setting['filter_size'], padding='same',
                                        name='Convolution'+(str)(index),
                                        input_shape=(seq_size, code_dim),
                                        use_bias=bias)(previous_layer)
            previous_layer = convolution
            if self.__add_batch_normalize:
                batch = BatchNormalization(name='BatchNormal'+(str)(index))(previous_layer)
                previous_layer = batch
            previous_layer = Activation('relu')(previous_layer)
        #add lstm rnn layer#
        lstm_forward = LSTM(self.__lstm_layer_number, return_sequences=True, activation='tanh',
                            name='LSTM_Forward', dropout=self.__dropout)(previous_layer)
        lstm_backword = LSTM(self.__lstm_layer_number, return_sequences=True,
                             go_backwards=True, activation='tanh',
                             name='LSTM_Backword', dropout=self.__dropout)(previous_layer)
        ####################
        #concatenate two lstm rnn layers
        concatenate_layer = concatenate(inputs=[lstm_forward, lstm_backword], name='Concatenate')
        #create last layer to predict the result
        if self.__output_dim == 1:
            last_activation = 'sigmoid'
        else:
            last_activation = 'softmax'
        last = Convolution1D(activation=last_activation, filters=self.__output_dim,
                             kernel_size=1, name='Result')(concatenate_layer)
        #create model
        model = SeqAnnModel(inputs=input_layers, outputs=last)
        #set optimizer metrics,and loss to the model
        optimizer = Adam(lr=self.__learning_rate)
        custom_metrics = []
        not_include_keys = ["loss", "SeqAnnModel"]
        for key,value in self.__custom_objects.items():
            if key not in not_include_keys:
                custom_metrics.append(value)
        model.compile(optimizer=optimizer, loss=self.__custom_objects['loss'],
                      metrics=custom_metrics)
        return model
    