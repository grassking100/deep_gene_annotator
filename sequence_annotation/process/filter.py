import torch
from torch import nn
from .customized_layer import BasicModel
from .rnn import ProjectedRNN

class RNNFilter(BasicModel):
    def __init__(self,in_channels,name=None,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        
        self.projected_rnn = ProjectedRNN(in_channels=in_channels,
                                          out_channels=self.out_channels,
                                          name=name,**kwargs)
                    
        if name is None or name=='':
            self.name = ''
            self._name=''
        else:
            self.name = "{}_".format(name)
            self._name=name
        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config['rnn'] = self.projected_rnn.get_config()
        config['name'] = self._name
        return config
        
    def forward(self,features,lengths,target_feature=None):
        if target_feature is None:
            target_feature = features
        gate = self.projected_rnn(features,lengths)
        result = target_feature * gate
        self.update_distribution(self.projected_rnn.saved_distribution)
        self.update_distribution(gate,'{}filter_gate'.format(self.name))
        self.update_distribution(result,'{}filter_result'.format(self.name))
        return result

class FilteredRNN(BasicModel):
    def __init__(self,in_channels,out_channels,num_layers=None,hidden_size=None,
                 filter_hidden_size=None,filter_num_layers=None,
                 output_act=None,name=None,**kwargs):
        super().__init__()
        if name is None or name=='':
            self.name = ''
            self.filter_name = 'filter'
            self.result_name = 'post_filter_rnn'  
        else:
            self.name = name
            self.filter_name = '{}_filter'.format(name)
            self.result_name = "{}_post_filter_rnn".format(self.name)
            
        if filter_num_layers is None:
            filter_num_layers = num_layers
        if filter_hidden_size is None:
            filter_hidden_size = hidden_size
            
        self.filter = RNNFilter(in_channels,hidden_size=filter_hidden_size,
                                num_layers=filter_num_layers,name=self.filter_name,
                                output_act='sigmoid',**kwargs)
        self.rnn = ProjectedRNN(in_channels,out_channels,hidden_size=hidden_size,
                                num_layers=num_layers,name=self.name,
                                output_act=output_act,**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config['filter'] = self.filter.get_config()
        config['rnn'] = self.rnn.get_config()
        config['name'] = self.name
        return config
        
    def forward(self,features,lengths,target_feature=None,**kwargs):
        features = self.filter(features,lengths,target_feature=target_feature)
        result = self.rnn(features,lengths)
        self.update_distribution(self.filter.saved_distribution)
        self.update_distribution(result,key=self.result_name)
        return result
