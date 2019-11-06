import torch
from torch import nn
from .customized_layer import BasicModel
from .cnn import Conv1d
from .rnn import GRU,ProjectedGRU

class RNNAttention(BasicModel):
    def __init__(self,in_channels,hidden_size=None,
                 name=None,use_softmax=True,on_site=False,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size or 16
        if on_site and use_softmax:
            raise Exception("The on_site and the use_softmax can be set to true on the same time")
        self.on_site = on_site
        self.use_softmax = use_softmax
        if self.on_site:
            self.out_channels = 1
        else:
            self.out_channels = in_channels
        
        self.projected_rnn = ProjectedGRU(in_channels=in_channels,hidden_size=self.hidden_size,
                                          out_channels=self.out_channels,**kwargs)
        
        if self.use_softmax:
            self.softmax_log = nn.LogSoftmax(dim=1)
        if name is None:
            self.name = ''
        else:
            self.name = "{}_".format(name)
        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config.update(self.projected_rnn.get_config())
        config['use_softmax'] = self.use_softmax
        config['on_site'] = self.on_site
        config['name'] = self.name
        return config
        
    def forward(self,features,lengths):
        attention_value = self.projected_rnn(features,lengths)
        if self.use_softmax:
            attention_gate = self.softmax_log(attention_value).exp()
        else:
            attention_gate = torch.sigmoid(attention_value)
        attention_result = features * attention_gate
        self._distribution['{}attention_value'.format(self.name)] = attention_value
        self._distribution['{}attention_gate'.format(self.name)] = attention_gate
        self._distribution['{}attention_result'.format(self.name)] = attention_result
        return attention_result

ATTENTION_LAYER={'RNNAttention':RNNAttention}
