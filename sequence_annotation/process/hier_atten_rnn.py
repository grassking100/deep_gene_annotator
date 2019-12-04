import torch
from torch import nn
from .customized_layer import BasicModel
from .cnn import Conv1d
from .rnn import GRU,ProjectedGRU
from .attention import RNNAttention

class AttenGRU(BasicModel):
    def __init__(self,in_channels,num_layers=None,out_channels=None,name=None,
                 hidden_size=None,atten_hidden_size=None,use_softmax=False,
                 atten_num_layers=None,**kwargs):
        super().__init__()
        if name is None or name=='':
            self.name = ''
            self.atten_name = 'atten'
            self.result_name = 'post_attention_rnn'  
        else:
            self.name = name
            self.atten_name = '{}_atten'.format(name)
            self.result_name = "{}_post_attention_rnn".format(self.name)
            
        if atten_num_layers is None:
            atten_num_layers = num_layers
        if atten_hidden_size is None:
            atten_hidden_size = hidden_size
            
        self.atten = RNNAttention(in_channels,hidden_size=atten_hidden_size,
                                  num_layers=atten_num_layers,name=self.atten_name,use_softmax=use_softmax,**kwargs)
        self.rnn = ProjectedGRU(in_channels,out_channels,hidden_size=hidden_size,num_layers=num_layers,
                                name=self.name,**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.reset_parameters()
        
    def forward(self,features,lengths,target_feature=None):
        features = self.atten(features,lengths,target_feature=target_feature)
        result = self.rnn(features,lengths)
        self.update_distribution(self.atten.saved_distribution)
        self.update_distribution(result,key=self.result_name)
        return result
        
    def get_config(self):
        config = super().get_config()
        config['atten'] = self.atten.get_config()
        config['rnn'] = self.rnn.get_config()
        config['name'] = self.name
        return config
        
class HierAttenGRU(BasicModel):
    def __init__(self,in_channels,use_first_atten=True,use_second_atten=True,
                 use_common_atten=False,use_softmax=False,
                 hidden_size=None,num_layers=None,
                 atten_hidden_size=None,atten_num_layers=None,
                 hier_option=None,out_channels=None,**kwargs):
        super().__init__()
        out_channels = out_channels or 2
        if out_channels%2 == 0:
            out_channels = int(out_channels/2)
        else:
            raise Exception() 
            
        self.in_channels = in_channels
        if use_first_atten:
            self.rnn_0 = AttenGRU(in_channels,out_channels=out_channels,name='first',
                                  use_softmax=use_softmax,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  atten_num_layers=atten_num_layers,
                                  atten_hidden_size=atten_hidden_size,**kwargs)
        else:
            self.rnn_0 = ProjectedGRU(in_channels,out_channels=out_channels,name='first',
                                      hidden_size=hidden_size,num_layers=num_layers,
                                      **kwargs)
            
        self.hier_option = hier_option or 'before_attention'
        if self.hier_option not in ['before_attention','after_attention','independent']:
            raise Exception("Wrong hier_option")
        if use_second_atten:
            self.rnn_1 = AttenGRU(in_channels,out_channels=out_channels,name='second',
                                  use_softmax=use_softmax,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  atten_num_layers=atten_num_layers,
                                  atten_hidden_size=atten_hidden_size,**kwargs)
        else:
             self.rnn_1 = ProjectedGRU(in_channels,out_channels=out_channels,name='second',
                                       hidden_size=hidden_size,num_layers=num_layers,**kwargs)
                
        if use_common_atten:
            self.common_atten = RNNAttention(in_channels,name='common_atten',
                                             use_softmax=use_softmax,
                                             num_layers=atten_num_layers,
                                             hidden_size=atten_hidden_size,**kwargs)
        self.use_common_atten = use_common_atten    
        self.out_channels = self.rnn_0.out_channels + self.rnn_1.out_channels
        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config['hier_option'] = self.hier_option
        config['rnn_0'] = self.rnn_0.get_config()
        config['rnn_1'] = self.rnn_1.get_config()
        if self.use_common_atten:
            config['common_atten'] = self.common_atten.get_config()
        return config
        
    def forward(self,x,lengths):
        if self.use_common_atten:
            common_atten_value = self.common_atten(x,lengths)
            x = common_atten_value
        result_0 = self.rnn_0(x,lengths)
        gated_result_0 = torch.sigmoid(result_0)
        if self.hier_option == 'independent':
            result_1 = self.rnn_1(x,lengths)
        else:
            gated_x = x*gated_result_0
            self.update_distribution(gated_x,key='gated_x')
            if self.hier_option == 'before_attention':
                result_1 = self.rnn_1(gated_x,lengths)
            else:
                result_1 = self.rnn_1(x,lengths,target_feature=gated_x)
        gated_result_1 = torch.sigmoid(result_1)
        result = torch.cat([gated_result_0,gated_result_1],1)
        if self.use_common_atten:
            self.update_distribution(common_atten_value,key='common_atten_value')
        self.update_distribution(self.rnn_0.saved_distribution)
        self.update_distribution(self.rnn_1.saved_distribution)
        self.update_distribution(result_0,key='result_0')
        self.update_distribution(gated_result_0,key='gated_result_0')
        self.update_distribution(result_1,key='result_1')
        self.update_distribution(gated_result_1,key='gated_result_1')
        self.update_distribution(result,key='gated_stack_result')
        return result
