import torch
from torch import nn
from .customized_layer import BasicModel
from .cnn import Conv1d
from .rnn import GRU,ProjectedGRU
from .attention import RNNAttention

class AttenGRU(BasicModel):
    def __init__(self,in_channels,hidden_size=None,num_layers=None,out_channels=None,
                 #use_softmax=False,on_site=False,
                 name=None,customized_gru_init=None,customized_cnn_init=None,**kwargs):
        super().__init__()
        self.name = name
        self.customized_gru_init = customized_gru_init
        self.customized_cnn_init = customized_cnn_init
        self.atten = RNNAttention(in_channels,hidden_size=hidden_size,num_layers=num_layers,
                                  #use_softmax=use_softmax,on_site=on_site,
                                  name=name,
                                  customized_gru_init=customized_gru_init,
                                  customized_cnn_init=customized_cnn_init,**kwargs)
        self.rnn = ProjectedGRU(in_channels,hidden_size,out_channels,num_layers=num_layers,
                                customized_gru_init=customized_gru_init,
                                customized_cnn_init=customized_cnn_init,**kwargs)
        self.result_name = 'post_attention_rnn'
        if self.name is not None:
            self.result_name = "{}_{}".format(self.name,self.result_name)
        self.reset_parameters()
        
    def forward(self,features,lengths):
        attention = self.atten(features,lengths)
        result = self.rnn(attention,lengths)
        self._distribution.update(self.atten.saved_distribution)
        self._distribution[self.result_name] = result
        return result
        
    def get_config(self):
        config = {}
        config['atten'] = self.atten.get_config()
        config['rnn'] = self.rnn.get_config()
        config['name'] = self.name
        config['customized_gru_init'] = self.customized_gru_init
        config['customized_cnn_init'] = self.customized_cnn_init
        return config
        
class HierAttenGRU(BasicModel):
    def __init__(self,in_channels,use_first_atten=True,use_second_atten=True,use_common_atten=False,
                  use_softmax=False,on_site=False,**kwargs):
        super().__init__()
        self.rnn_0 = AttenGRU(in_channels,out_channels=1,name='first',
                              use_attention=use_first_atten,
                              use_softmax=use_softmax,on_site=on_site,
                              **kwargs)
        self.rnn_1 = AttenGRU(in_channels,out_channels=1,name='second',
                              use_attention=use_second_atten,
                              use_softmax=use_softmax,on_site=on_site,
                              **kwargs)
        if use_common_atten:
            self.common_atten = RNNAttention(in_channels,name='common_atten',
                                             use_softmax=use_softmax,on_site=on_site,
                                             **kwargs)
        self.use_first_atten = use_first_atten
        self.use_second_atten = use_second_atten
        self.use_common_atten = use_common_atten    
        self.out_channels = 2
        self.reset_parameters()
        
    def get_config(self):
        config = dict(self.rnn_0.get_config())
        config['use_first_atten'] = self.use_first_atten
        config['use_second_atten'] = self.use_second_atten
        config['use_common_atten'] = self.use_common_atten
        return config
        
    def forward(self,x,lengths):
        if self.use_common_atten:
            common_atten_value = self.common_atten(x,lengths)
            x = common_atten_value
        feature_for_gate_0 = feature_for_gate_1 = x
        result_0 = self.rnn_0(feature_for_gate_0,lengths)
        gated_result_0 = torch.sigmoid(result_0)
        gated_x = feature_for_gate_1*gated_result_0
        result_1 = self.rnn_1(gated_x,lengths)
        gated_result_1 = torch.sigmoid(result_1)
        result = torch.cat([gated_result_0,gated_result_1],1)
        if self.use_common_atten:
            self._distribution['common_atten_value'] = common_atten_value
        self._distribution.update(self.rnn_0.saved_distribution)
        self._distribution.update(self.rnn_1.saved_distribution)
        self._distribution['result_0'] = result_0
        self._distribution['gated_result_0'] = gated_result_0
        self._distribution['gated_x'] = gated_x
        self._distribution['result_1'] = result_1
        self._distribution['gated_result_1'] = gated_result_1
        self._distribution['gated_stack_result'] = result
        return result

class GatedStackGRU(BasicModel):
    def __init__(self,in_channels,hidden_size,num_layers=None,**kwargs):
        super().__init__()
        self.num_layers = num_layers or 1
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.projected_rnn_0 = ProjectedGRU(in_channels=in_channels,hidden_size=hidden_size,
                                          out_channels=1,num_layers=self.num_layers,**kwargs)
        
        self.projected_rnn_1 = ProjectedGRU(in_channels=in_channels,hidden_size=hidden_size,
                                          out_channels=1,num_layers=self.num_layers,**kwargs)
        self.out_channels = 2
        
    def get_config(self):
        config = dict(self.projected_rnn_0.get_config())
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['hidden_size'] = self.hidden_size
        config['num_layers'] = self.num_layers
        return config
        
    def forward(self,x,lengths):
        if isinstance(x,list):
            feature_for_gate_0, feature_for_gate_1 = x
        else:
            feature_for_gate_0 = feature_for_gate_1 = x
        
        result_0,post_rnn_0 = self.projected_rnn_0(feature_for_gate_0,lengths,return_intermediate=True)
        gated_result_0 = torch.sigmoid(result_0)
        gated_x = feature_for_gate_1*gated_result_0
        result_1,post_rnn_1 = self.projected_rnn_1(gated_x,lengths,return_intermediate=True)
        gated_result_1 = torch.sigmoid(result_1)
        result = torch.cat([gated_result_0,gated_result_1],1)

        self._distribution['post_rnn_0'] = post_rnn_0
        self._distribution['result_0'] = result_0
        self._distribution['gated_result_0'] = gated_result_0
        self._distribution['gated_x'] = gated_x
        self._distribution['post_rnn_1'] = post_rnn_1
        self._distribution['result_1'] = result_1
        self._distribution['gated_result_1'] = gated_result_1
        self._distribution['gated_stack_result'] = result
        return result
