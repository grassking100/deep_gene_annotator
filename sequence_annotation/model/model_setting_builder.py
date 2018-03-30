from . import Builder
class ModelSettingBuilder(Builder):
    def __init__(self,input_dim,output_dim):
        self._setting={}
        self._setting['global']={}
        self._setting['layer']={}
        self._add_global_setting(input_dim,output_dim)
        self._output_dim = None
        self._dropout = None
    def _add_global_setting(self,input_dim,output_dim):
        self._setting['global']['input_dim']=input_dim
        self._setting['global']['output_dim']=output_dim
    def _add_layer(self,name,dropout,batch_normalize):
        setting = {}
        setting['name'] = name
        setting['dropout'] = dropout
        setting['batch_normalize'] = batch_normalize
        return setting
    def add_CNN_(self,number,shape,name=None,dropout=0,batch_normalize=False,layer_subtype="1D"):
        setting = self._add_layer(name=name,dropout=dropout,batch_normalize=batch_normalize)
        setting['number']=number
        setting['shape']=shape
        setting['layer_type']="CNN_"+layer_subtype
        self._setting['layer'].append(setting)
    def add_RNN(self,number,name=None,dropout=0,batch_normalize=False,layer_subtype="LSTM"):
        setting = self._add_layer(name=name,dropout=dropout,batch_normalize=batch_normalize)
        setting['number']=number
        setting['layer_type']="RNN_"+layer_subtype
        self._setting['layer'].append(setting)
    def _validate(self):
        pass
    def build(self):
        self._validate()
        return self._setting
