"""This submodule contains class which can create model"""
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.engine.training import Model
from keras.layers import RNN,Input, Convolution1D, Dense
from keras.layers import LSTM, Activation,Bidirectional
from keras.layers import Conv2DTranspose
from . import Builder
class ModelBuilder(Builder):
    """This class will create and return sequence annotation model"""
    def __init__(self,model_setting):
        self.setting = model_setting
        self._layers = {}
        self._input_layer_ids =[]
        self._output_layer_ids =[]
    def _build_layers(self):
        for name,setting in self.setting['layer'].items():
            setting['keras_setting']['name'] = name
            self._build_layer(setting)
    def _build_layer(self,setting):
        layer_type = setting['type']
        if layer_type=='CNN_1D':
            layer = self._build_CNN_1D(setting)
        elif layer_type=='Input':
            layer = self._build_input(setting)
        else:
            try:
                exec('from keras.layers import {layer_type}'.format(layer_type=layer_type))
                exec('self._temp_layer_class={layer_type}'.format(layer_type=layer_type))
                layer = self._temp_layer_class(**setting['keras_setting'])
                if isinstance(self._temp_layer_class,RNN) and setting['bidirection_setting']!=None:
                    layer = self._to_bidirectional(layer,setting)                    
            except ImportError as e:
                raise Exception("Layer,{layer},has not implement yet".format(layer=layer_type))
        layer_name = setting['keras_setting']['name']
        if not layer_name in self._layers.keys():
            self._layers[layer_name] = {}
            self._layers[layer_name]['layer'] = layer
            self._layers[layer_name]['setting'] = setting
            if setting['is_output']:
                self._output_layer_ids.append(layer_name)
            if layer_type=='Input':
                self._layers[layer_name]['is_linked'] = True
                self._input_layer_ids.append(layer_name)
            else:
                self._layers[layer_name]['is_linked'] = False
        else:
            raise Exception("Duplicate name:"+layer_name)
    def _link_layers(self):
        for name,setting in self.setting['layer'].items():
            setting['name'] = name
            self._link_layer(setting)
    def _link_layer(self, setting):
        present_layer_status = self._layers[setting['keras_setting']['name']]
        if not present_layer_status['is_linked']:
            if setting['previous_layer'] is not None:
                previous_layer_status = self._layers[setting['previous_layer']]
                if not previous_layer_status['is_linked']:
                    self._link_layer(previous_layer_status['setting'])
                present_layer = present_layer_status['layer']
                previous_layer = previous_layer_status['layer']
                present_layer_status['layer'] = present_layer(previous_layer)
                present_layer_status['is_linked'] = True
    def _build_model(self):
        input_layers = []
        output_layers = []
        for id_ in self._input_layer_ids:
            input_layers.append(self._layers[id_]['layer'])
        for id_ in self._output_layer_ids:
            output_layers.append(self._layers[id_]['layer'])
        return Model(inputs=input_layers, outputs=output_layers)
    def build(self):
        """Create and return model"""
        self._validate()
        self._build_layers()
        self._link_layers()
        return self._build_model()
    def _validate(self):
        pass
    def _build_CNN_1D(self,setting):
        return_layer = Convolution1D(**setting['keras_setting'])
        return return_layer
    def _to_bidirectional(self,inner_layers,setting):
        return_layer = Bidirectional(inner_layers,**setting['bidirection_setting'])
        return_layer.name = inner_layers.name
        return return_layer
    def _build_input(self,setting):
        setting = setting['keras_setting']
        setting['shape'] = tuple(setting['shape'])
        return_layer = Input(**setting)
        return return_layer
    """def _build_concatenate(self,previous_layers):
        return concatenate(inputs=previous_layers, name='Concatenate')"""