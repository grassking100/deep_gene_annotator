"""This submodule contains class which can create model"""
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.engine.training import Model
from keras.layers import Input, Convolution1D, Dense
from keras.layers import LSTM, Activation,Bidirectional
from keras.layers import Conv2DTranspose
from . import Builder
from . import AttrValidator
class ModelBuilder(Builder):
    """This class will create and return sequence annotation model"""
    def __init__(self,model_setting):
        self.setting = model_setting
        self._layer_index = 1
    def build(self):
        """Create and return model"""
        self._validate()
        
        for setting in self.setting['layer']:
            layer_type = setting['type']
            if layer_type=='CNN_1D':
                previous_layer = self._add_CNN_1D(previous_layer,setting)
            elif layer_type=='Dense':
                previous_layer = self._add_Dense(previous_layer,setting)
            elif layer_type=='RNN_LSTM':
                previous_layer = self._add_LSTM(previous_layer,setting)
            elif layer_type=='RNN_Bi_LSTM':
                previous_layer = self._add_LSTM(previous_layer,setting)
                previous_layer = self.to_bidirectional(previous_layer)
            elif layer_type=='Input':
                input_layer = self._add_input(setting)
                previous_layer = input_layer
            else:
                raise Exception("Layer,{layer},has not implment yet".format(layer=layer_type))
        #last_layer = self._add_output(previous_layer)
        model = Model(inputs=input_layer, outputs=previous_layer)
        return model
    def _add_Dense(self,previous_layer,layer_setting):
        layer = Dense(units=layer_setting['number'],
                      activation=layer_setting['activation'],
                      name=self._get_new_name(layer_setting['name']))
        next_layer = layer(previous_layer)
        return next_layer
    def _add_CNN_2D_transpose(self,previous_layer,layer_setting):
        layer = Conv2DTranspose(filters=layer_setting['number'],
                                kernel_size=layer_setting['shape'],
                                padding='same',name=self._get_new_name(layer_setting['name']))
        next_layer = layer(previous_layer)
        return next_layer
    def _get_new_name(self,name):
        new_name = name+"_"+str(self._layer_index)
        self._layer_index += 1
        return new_name
    def _validate(self):
        attr_validator = AttrValidator(self,False,True,False,None)
        attr_validator.validate()
    def _add_CNN_1D(self,previous_layer,layer_setting):
        next_layer = Convolution1D(filters=layer_setting['number'],
                                   kernel_size=layer_setting['shape'],
                                   padding='same',
                                   name=self._get_new_name(layer_setting['name']),
                                   use_bias=not layer_setting['batch_normalize'])(previous_layer)
        if layer_setting['batch_normalize']:
            batch = BatchNormalization(name=self._get_new_name('BatchNormal'))(next_layer)
            next_layer = Activation('relu',name=self._get_new_name('ReLu'))(batch)
        return next_layer
    def to_bidirectional(self,layer):
        return_layer = Bidirectional(layer,merge_mode='concat')
        return_layer.name = "Bidirection_"+layer.name
        return return_layer
    def _add_LSTM(self,previous_layer,layer_setting):
        inner_layer =LSTM(layer_setting['number'],return_sequences=True,
                         activation='tanh',name=self._get_new_name(layer_setting['name']),
                         dropout = self.setting['global']['dropout'])
        if layer_setting['bidirection']:
            next_layer = self.to_bidirectional(inner_layer)(previous_layer)
        else:
            next_layer = inner_layer(previous_layer)
        return next_layer
    def _add_input(self,layer_setting):
        input_shape = tuple(layer_setting['number'])
        print(input_shape)
        next_layer = Input(shape=input_shape, name=layer_setting['name'])
        return next_layer
    """def _add_output(self,previous_layer):
        dim = tuple(self.setting['global']['output_dimension'])
        if  dim== 1:
            last_activation = 'sigmoid'
        else:
            last_activation = 'softmax'
        next_layer = Convolution1D(activation=last_activation, filters=dim,
                                   kernel_size=1, name='Output')(previous_layer)
        return next_layer"""
    def _add_concatenate(self,previous_layers):
        return concatenate(inputs=previous_layers, name='Concatenate')