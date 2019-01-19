"""This submodule contains class which can create model"""
from keras.engine.training import Model
from keras.layers import RNN,Input
from keras.layers import Bidirectional,Permute
import copy
from .block_layer_builder import Cnn1dBatchReluBuilder,ResidualLayerBuilder,DeepResidualLayerBuilder
from .block_layer_builder import DeepDenseLayerBuilder,DenseLayerBuilder

def set_regularizer(regularizer_type,regularizer):
    temp = None
    command = 'from keras.regularizers import {regularizer_type}'.format(regularizer_type=regularizer_type)
    exec(command)
    command = 'temp={regularizer}'.format(regularizer=regularizer)
    exec(command)
    return temp

class ModelBuilder:
    """This class will create and return sequence annotation model"""
    def __init__(self,model_setting):
        self.setting = model_setting
        self._layers = {}
        self._input_layer_ids =[]
        self._output_layer_ids =[]

    def _build_layers(self):
        for setting in self.setting['layer'].values():
            self._build_layer(setting)

    def _get_layer(self,setting):
        layer_type = setting['type']
        keras_setting = setting['keras_setting']
        if 'kernel_regularizer' in keras_setting.keys():
            regularizer = keras_setting['kernel_regularizer']
            regularizer_type =regularizer[:2]
            temp = set_regularizer(regularizer_type,regularizer)
            keras_setting['kernel_regularizer'] = temp
        if 'bias_regularizer' in keras_setting.keys():
            regularizer = keras_setting['bias_regularizer']
            regularizer_type =regularizer[:2]
            temp = set_regularizer(regularizer_type,regularizer)
            keras_setting['bias_regularizer'] = temp
        if 'recurrent_regularizer' in keras_setting.keys():
            regularizer = keras_setting['recurrent_regularizer']
            regularizer_type =regularizer[:2]
            temp = set_regularizer(regularizer_type,regularizer)
            keras_setting['recurrent_regularizer'] = temp
        if 'activity_regularizer' in keras_setting.keys():
            regularizer = keras_setting['activity_regularizer']
            regularizer_type =regularizer[:2]
            temp = set_regularizer(regularizer_type,regularizer)
            keras_setting['activity_regularizer'] = temp
        if layer_type == 'Input':
            layer = self._build_input(setting)
        elif layer_type == 'Cnn1dBatchRelu':
            layer = self._build_Cnn1dBatchRelu(setting)
        elif layer_type == 'ResidualLayer':
            layer = self._build_ResidualLayer(setting)
        elif layer_type == 'DeepResidualLayer':
            layer = self._build_DeepResidualLayerBuilder(setting)
        elif layer_type == 'DenseLayer':
            layer = self._build_DenseLayerBuilder(setting)
        elif layer_type == 'DeepDenseLayer':
            layer = self._build_DeepDenseLayerBuilder(setting)
        elif layer_type == 'Permute':
            layer = self._build_Permute(setting)
        else:
            try:
                exec('from keras.layers import {layer_type}'.format(layer_type=layer_type))
                exec('self._temp_layer_class={layer_type}'.format(layer_type=layer_type))
                layer = self._temp_layer_class(**keras_setting)
            except ImportError as e:
                raise Exception("Layer,{layer},has not been implemented yet".format(layer=layer_type))
        is_RNN = isinstance(layer.__class__,RNN.__class__)
        if  is_RNN and 'bidirection_setting' in setting.keys():
            layer = self._to_bidirectional(layer,setting)
        return layer

    def _build_layer(self,setting):
        if not setting['name'] in self._layers.keys():
            setting_ = copy.deepcopy(setting)
            layer = self._get_layer(setting_)
            layer_type = setting_['type']
            layer_name = setting_['name']
            self._layers[layer_name] = {}
            self._layers[layer_name]['layer'] = layer
            self._layers[layer_name]['setting'] = setting
            if 'is_output' not in setting_:
                setting_['is_output'] = False
            if setting_['is_output']:
                self._output_layer_ids.append(layer_name)
            if layer_type=='Input':
                self._layers[layer_name]['is_linked'] = True
                self._input_layer_ids.append(layer_name)
            else:
                self._layers[layer_name]['is_linked'] = False
        else:
            raise Exception("Duplicate name:"+layer_name)

    def _link_layers(self):
        for setting in self.setting['layer'].values():
            self._link_layer(setting)

    def _link_layer(self, setting):
        present_layer_status = self._layers[setting['name']]
        if not present_layer_status['is_linked']:
            if setting['previous_layer'] is not None:
                present_layer = present_layer_status['layer']
                previous_layers = []
                previous_layer_ids = setting['previous_layer']
                if not isinstance(previous_layer_ids,list):
                    previous_layer_statuses = [self._layers[previous_layer_ids]]
                else:
                    previous_layer_statuses = [self._layers[id_] for id_ in previous_layer_ids]
                for previous_layer_status in previous_layer_statuses:
                    if not previous_layer_status['is_linked']:
                        self._link_layer(previous_layer_status['setting'])
                    previous_layers.append(previous_layer_status['layer'])
                if len(previous_layers)==1:
                    present_layer_status['layer'] = present_layer(previous_layers[0])
                else:
                    present_layer_status['layer'] = present_layer(previous_layers)
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
        for name,setting in self.setting['layer'].items():
            setting['name'] = name
            if "name" not in setting["keras_setting"].keys():
                setting["keras_setting"]['name']=name
        self._build_layers()
        self._link_layers()
        model = self._build_model()
        return model

    def _validate(self):
        previous_layers = []
        for layer in self.setting['layer'].values():
            previous_layer = layer['previous_layer']
            if previous_layer is not None:
                if isinstance(previous_layer,list):
                    previous_layers += previous_layer
                else:
                    previous_layers += [previous_layer]
        previous_layers= set(previous_layers)
        if len(self.setting['layer']) != (len(previous_layers)+1):
            raise Exception("Some layer is missing,please check the model setting!!!")

    def _to_bidirectional(self,inner_layers,setting):
        return_layer = Bidirectional(inner_layers,**setting['bidirection_setting'])
        return_layer.name = inner_layers.name
        return return_layer

    def _build_input(self,setting):
        setting = setting['keras_setting']
        setting['shape'] = tuple(setting['shape'])
        return_layer = Input(**setting)
        return return_layer

    def _build_Permute(self,setting):
        setting = setting['keras_setting']
        setting['dims'] = tuple(setting['dims'])
        return_layer = Permute(**setting)
        return return_layer

    def _build_Cnn1dBatchRelu(self,setting):
        setting['setting']['name'] = setting['name']
        return_layer = Cnn1dBatchReluBuilder().build(**setting['setting'])
        return return_layer

    def _build_ResidualLayer(self,setting):
        setting['setting']['name'] = setting['name']
        return_layer = ResidualLayerBuilder().build(**setting['setting'])
        return return_layer

    def _build_DeepResidualLayerBuilder(self,setting):
        setting['setting']['name'] = setting['name']
        return_layer = DeepResidualLayerBuilder().build(**setting['setting'])
        return return_layer

    def _build_DenseLayerBuilder(self,setting):
        setting['setting']['name'] = setting['name']
        return_layer = DenseLayerBuilder().build(**setting['setting'])
        return return_layer

    def _build_DeepDenseLayerBuilder(self,setting):
        setting['setting']['name'] = setting['name']
        return_layer = DeepDenseLayerBuilder().build(**setting['setting'])
        return return_layer
