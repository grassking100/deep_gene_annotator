"""This submodule contains class which can create model"""
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.engine.training import Model
from keras.layers import Input, Convolution1D
from keras.layers import LSTM, Activation
from . import Builder
from . import AttrValidator
class ModelBuilder(Builder):
    """This class will create and return sequence annotation model"""
    def __init__(self,model_setting):
        self._setting = model_setting
    def build(self):
        """Create and return model"""
        self._validate()
        input_layer = self._add_input()
        previous_layer = input_layer
        for setting in self._setting['layer']:
            layer_type = setting['layer_type']
            if layer_type=='CNN_1D':
                previous_layer = self._add_CNN_1D(previous_layer,setting)
            elif layer_type=='RNN_LSTM':
                previous_layer = self._add_LSTM(previous_layer,setting)
                previous_layer = self.to_bidirectional(previous_layer)
            else:
                raise Exception("Layer,{layer},has not implment yet".format(layer=layer_type))
        last_layer = self._add_output(previous_layer)
        model = Model(inputs=input_layer, outputs=last_layer)
        return model
    def _validate(self):
        attr_validator = AttrValidator(self,False,True,False,None)
        attr_validator.validate()
    def _add_CNN_1D(self,previous_layer,layer_setting):
        next_layer = Convolution1D(filters=layer_setting['number'],
                                   kernel_size=layer_setting['shape'],
                                   padding='same',name=layer_setting['name'],
                                   input_shape=(None, self._setting['global']['input_dim']),
                                   use_bias=not layer_setting['batch_normalize'])(previous_layer)
        if layer_setting['batch_normalize']:
            batch = BatchNormalization(name='BatchNormal'+(str)(index))(next_layer)
            next_layer = Activation('relu',name='ReLu'+(str)(index))(batch)
        return next_layer
    def to_bidirectional(self,layers):
        return Bidirectional(layers)
    def _add_LSTM(self,previous_layer,layer_setting):
        next_layer =LSTM(layer_setting['number'],
                         return_sequences=True,
                         activation='tanh',
                         layer_setting['name'],
                         layer_setting['dropout'])(previous_layer)
        return next_layer
    def _add_input(self):
        seq_input_shape = (None, self._setting['global']['input_dim'])
        next_layer = Input(shape=seq_input_shape, name='Input')
        return next_layer
    def _add_output(self,previous_layer):
        dim = self._setting['global']['output_dim']
        if  dim== 1:
            last_activation = 'sigmoid'
        else:
            last_activation = 'softmax'
        next_layer = Convolution1D(activation=last_activation, filters=dim,
                                   kernel_size=1, name='Onput')(previous_layer)
        return next_layer
    def _add_concatenate(self,previous_layers)
        return concatenate(inputs=previous_layers, name='Concatenate')