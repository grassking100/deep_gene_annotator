"""This submodule contains class which can create model"""
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.engine.training import Model
from keras.layers import Input, Convolution1D
from keras.layers import LSTM, Activation
from . import SeqAnnModel
from . import Builder
from . import AttrValidator
class SeqAnnModelBuilder(Builder):
    """This class will create and return sequence annotation model"""
    def __init__(self):
        self._cnn_setting = None
        self._lstm_layer_number = None
        self._add_batch_normalize = None
        self._output_dim = None
        self._dropout = None
    @property
    def cnn_setting(self):
        """Get convolution layer setting"""
        return self._cnn_setting
    @property
    def lstm_layer_number(self):
        """Get lstm layer number"""
        return self._lstm_layer_number
    @property
    def add_batch_normalize(self):
        """Get boolean about decide to use batch normalize or not"""
        return self._add_batch_normalize
    @property
    def dropout(self):
        """Get dropout"""
        return self._dropout
    @property
    def output_dim(self):
        """Get output dimention size"""
        return self._output_dim
    @cnn_setting.setter
    def cnn_setting(self, cnn_setting):
        """Set convolution layer setting"""
        self._cnn_setting = cnn_setting
        return self
    @lstm_layer_number.setter
    def lstm_layer_number(self, lstm_layer_number):
        """Set lstm layer number"""
        self._lstm_layer_number = lstm_layer_number
        return self
    @add_batch_normalize.setter
    def add_batch_normalize(self, add_batch_normalize):
        """Decide to use batch normalize or not"""
        self._add_batch_normalize = add_batch_normalize
        return self
    @dropout.setter
    def dropout(self, dropout):
        """Set dropout"""
        self._dropout = dropout
        return self
    @output_dim.setter
    def output_dim(self, output_dim):
        """Set output dimention size"""
        self._output_dim = output_dim
        return self
    def _validate(self):
        attr_validator = AttrValidator(self,False,True,False,None)
        attr_validator.validate()
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
        for index, setting in enumerate(self.cnn_setting):
            bias = not self.add_batch_normalize
            convolution = Convolution1D(filters=setting['filter_num'],
                                        kernel_size=setting['filter_size'], padding='same',
                                        name='Convolution'+(str)(index),
                                        input_shape=(seq_size, code_dim),
                                        use_bias=bias)(previous_layer)
            previous_layer = convolution
            if self.add_batch_normalize:
                batch = BatchNormalization(name='BatchNormal'+(str)(index))(previous_layer)
                previous_layer = batch
            previous_layer = Activation('relu',name='ReLu'+(str)(index))(previous_layer)
        #add lstm rnn layer#
        lstm_forward = LSTM(self.lstm_layer_number, return_sequences=True, activation='tanh',
                            name='LSTM_Forward', dropout=self.dropout)(previous_layer)
        lstm_backword = LSTM(self.lstm_layer_number, return_sequences=True,
                             go_backwards=True, activation='tanh',
                             name='LSTM_Backword', dropout=self.dropout)(previous_layer)
        ####################
        #concatenate two lstm rnn layers
        concatenate_layer = concatenate(inputs=[lstm_forward, lstm_backword], name='Concatenate')
        #create last layer to predict the result
        if self.output_dim == 1:
            last_activation = 'sigmoid'
        else:
            last_activation = 'softmax'
        last = Convolution1D(activation=last_activation, filters=self.output_dim,
                             kernel_size=1, name='Result')(concatenate_layer)
        #create model
        model = Model(inputs=input_layers, outputs=last)
        return model
    