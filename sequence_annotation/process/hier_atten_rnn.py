import torch
from torch import nn
from .customized_layer import Conv1d, BasicModel
from .rnn import GRU
from .attention import RNNAttention

class ProjectedGRU(BasicModel):
    def __init__(self,in_channels,hidden_size,out_channels,num_layers=None,**kwargs):
        super().__init__()
        self.num_layers = num_layers or 1
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.rnn = GRU(in_channels=self.in_channels,bidirectional=True,
                       hidden_size=self.hidden_size,batch_first=True,
                       num_layers=self.num_layers)
        self.project = Conv1d(in_channels=self.rnn.out_channels,
                              out_channels=self.out_channels,kernel_size=1)
        
    def get_config(self):
        config = {}
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['hidden_size'] = self.hidden_size
        config['num_layers'] = self.num_layers
        return config
        
    def forward(self,x,lengths):
        post_rnn = self.rnn(x,lengths)
        result,lengths,_ = self.project(post_rnn,lengths)
        return result

class AttenGRU(BasicModel):
    def __init__(self,in_channels,hidden_size=None,num_layers=None,out_channels=None,
                 use_softmax=False,on_site=False,name=None,**kwargs):
        super().__init__()
        self.name = name
        self.atten = RNNAttention(in_channels,hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  use_softmax=use_softmax,
                                  on_site=on_site,name=name)
        self.rnn = ProjectedGRU(in_channels,hidden_size,out_channels,num_layers=num_layers,**kwargs)
        self.result_name = 'post_attention_rnn'
        if self.name is not None:
            self.result_name = "{}_{}".format(self.name,self.result_name)
        
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
        return config
        
class HierAttenGRU(BasicModel):
    def __init__(self,in_channels,**kwargs):
        super().__init__()
        self.rnn_0 = AttenGRU(in_channels,out_channels=1,name='first',**kwargs)
        self.rnn_1 = AttenGRU(in_channels,out_channels=1,name='second',**kwargs)
        
    def get_config(self):
        config = dict(self.rnn_0.get_config())
        return config
        
    def forward(self,x,lengths):
        feature_for_gate_0 = feature_for_gate_1 = x
        result_0 = self.rnn_0(feature_for_gate_0,lengths)
        gated_result_0 = torch.sigmoid(result_0)
        gated_x = feature_for_gate_1*gated_result_0
        result_1 = self.rnn_1(gated_x,lengths)
        gated_result_1 = torch.sigmoid(result_1)
        result = torch.cat([gated_result_0,gated_result_1],1)        
        self._distribution.update(self.rnn_0.saved_distribution)
        self._distribution.update(self.rnn_1.saved_distribution)
        self._distribution['result_0'] = result_0
        self._distribution['gated_result_0'] = gated_result_0
        self._distribution['gated_x'] = gated_x
        self._distribution['result_1'] = result_1
        self._distribution['gated_result_1'] = gated_result_1
        self._distribution['gated_stack_result'] = result
        return result
