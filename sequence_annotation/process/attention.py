import torch
from torch import nn
from .customized_layer import Conv1d, BasicModel
from .rnn import GRU

class RNNAttention(BasicModel):
    def __init__(self,in_channels,hidden_size=None,num_layers=None,mode=None,
                 name=None,use_softmax=True,on_site=False):
        super().__init__()
        self.num_layers = num_layers or 1
        self.in_channels = in_channels
        self.hidden_size = hidden_size or 16
        if on_site and use_softmax:
            raise Exception("The on_site and the use_softmax can be set to true on the same time")
        self.on_site = on_site
        self.use_softmax = use_softmax
        self.rnn = GRU(in_channels=in_channels,bidirectional=True,
                       hidden_size=self.hidden_size,batch_first=True,
                       num_layers=self.num_layers)
        if self.on_site:
            self.out_channels = 1
        else:
            self.out_channels = in_channels
            
        self.project = Conv1d(in_channels=self.rnn.out_channels,out_channels=self.out_channels,kernel_size=1)
        
        
        if self.use_softmax:
            self.softmax_log = nn.LogSoftmax(dim=1)
        if name is None:
            self.name = ''
        else:
            self.name = "{}_".format(name)
        
    def get_config(self):
        config = {}
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['hidden_size'] = self.hidden_size
        config['num_layers'] = self.num_layers
        config['use_softmax'] = self.use_softmax
        config['on_site'] = self.on_site
        return config
        
    def forward(self,features,lengths):
        rnn_result = self.rnn(features,lengths)
        attention_value = self.project(rnn_result)[0]
        if self.use_softmax:
            attention_gate = self.softmax_log(attention_value).exp()
        else:
            attention_gate = torch.sigmoid(attention_value)
        attention_result = features * attention_gate
        self._distribution['{}attention_value'.format(self.name)] = attention_value
        self._distribution['{}attention_gate'.format(self.name)] = attention_gate
        self._distribution['{}attention_result'.format(self.name)] = attention_result
        return attention_result


class BiRNNAttention(BasicModel):
    def __init__(self,in_channels,mode=None,**kwargs):
        super().__init__()
        mode = mode or 'same'
        if mode not in ['first','second','same','each']:
            raise Exception("Get invalid mode {}".format(mode))
        self._mode = mode
        if mode == 'each':
            self.atten = RNNAttention(in_channels,name='first',**kwargs)
            self.atten_2 = RNNAttention(in_channels,name='second',**kwargs)
        else:
            self.atten = RNNAttention(in_channels,**kwargs)
        
    def get_config(self):
        config = dict(self.atten.get_config())
        config['mode'] = self._mode
        return config
        
    def forward(self,features,lengths):
        attention_result = self.atten(features,lengths)
        self._distribution.update(self.atten.saved_distribution)
        if self._mode ==  'each':
            attention_result_2 = self.atten_2(features,lengths)
            self._distribution.update(self.atten_2.saved_distribution)
            return [attention_result,attention_result_2]
        elif self._mode ==  'first':
            return [attention_result,features]
        elif self._mode ==  'second':
            return [features,attention_result]
        else:
            return [attention_result,attention_result]

        
ATTENTION_LAYER={'RNNAttention':RNNAttention,'BiRNNAttention':BiRNNAttention}