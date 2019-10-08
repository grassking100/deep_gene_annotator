from abc import abstractmethod
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
from .customized_layer import BasicModel,Concat
from .cnn import Conv1d
        
def customized_init_gru(rnn):
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            w_ir, w_iz, w_in = getattr(rnn,'weight_ih_l{}{}'.format(index,suffix)).chunk(3, 0)
            w_hr, w_hz, w_hn = getattr(rnn,'weight_hh_l{}{}'.format(index,suffix)).chunk(3, 0)
            ri_bias,zi_bias,ni_bias = getattr(rnn,'bias_ih_l{}{}'.format(index,suffix)).chunk(3, 0)
            rh_bias,zh_bias,nh_bias = getattr(rnn,'bias_hh_l{}{}'.format(index,suffix)).chunk(3, 0)
            constant_(ri_bias,-4)
            constant_(rh_bias,-4)
            constant_(zi_bias,4)
            constant_(zh_bias,4)
            constant_(ni_bias,0)
            constant_(nh_bias,0)
        
def _forward(rnn,x,lengths,state=None,batch_first=True):
    #Input:N,C,L, Output: N,C,L
    if not batch_first:
        x = x.transpose(0,1)    
    x = x.transpose(1,2)
    x = pack_padded_sequence(x,lengths, batch_first=batch_first)
    x = rnn(x,state)[0]
    x,_ = pad_packed_sequence(x, batch_first=batch_first)
    x = x.transpose(1,2)
    if not batch_first:
        x = x.transpose(0,1)
    return x

class _RNN(BasicModel):
    def __init__(self,in_channels,train_init_value=False,init_value=None,customized_init=None,**rnn_setting):
        super().__init__()
        self.customized_init = customized_init
        self.rnn = self._create_rnn(in_channels,**rnn_setting)
        self.rnn_setting = rnn_setting
        self.in_channels = in_channels
        self.hidden_size = self.rnn.hidden_size
        self.out_channels = self.rnn.hidden_size
        self.train_init_value = train_init_value
        self.init_value = init_value or 0
        
        if hasattr(self.rnn,'bidirectional') and self.rnn.bidirectional:
            direction_num = 2
            self.out_channels *= 2
        else:
            direction_num = 1
        val = [self.init_value]*self.rnn.hidden_size
        num_layers = self.rnn.num_layers if hasattr(self.rnn,'num_layers') else 1
        val = torch.Tensor(val)
        val = val.unsqueeze(0)
        val = val.repeat(num_layers*direction_num,1)
        self.init_states = torch.nn.Parameter(val,requires_grad=train_init_value)
        self.batch_first = self.rnn.batch_first if hasattr(self.rnn,'batch_first') else True
        self.reset_parameters()

    @abstractmethod
    def _create_rnn(self,in_channels,**rnn_setting):
        pass
    
    def get_config(self):
        config = {}
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['train_init_value'] = self.train_init_value
        config['init_value'] = self.init_value
        config['rnn_setting'] = self.rnn_setting
        config['customized_init'] = str(self.customized_init)
        return config
    
class GRU(_RNN):
    def _create_rnn(self,in_channels,**rnn_setting):
        return nn.GRU(in_channels,**rnn_setting)
    
    def forward(self,x,lengths=None,state=None):
        if lengths is None:
            lengths = [x.shape[2]]*len(x)
        if state is None:
            state = self.init_states.unsqueeze(1).repeat(1,len(x),1)
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x

    def reset_parameters(self):
        super().reset_parameters()
        if self.customized_init is not None:
            self.customized_init(self.rnn)
    
class LSTM(_RNN):
    def _create_rnn(self,in_channels,**rnn_setting):
        return nn.LSTM(in_channels,**rnn_setting)
    
    def forward(self,x,lengths,state=None):
        if state is None:
            state = self.init_states.repeat(len(x),1).cuda()
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x

class GatedStackGRU(BasicModel):
    def __init__(self,in_channels,hidden_size,num_layers=None,**kwargs):
        super().__init__()
        self.num_layers = num_layers or 1
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.rnn_0 = GRU(in_channels=in_channels,bidirectional=True,
                         hidden_size=hidden_size,batch_first=True,
                         num_layers=self.num_layers)
        self.rnn_1 = GRU(in_channels=in_channels,bidirectional=True,
                         hidden_size=hidden_size,batch_first=True,
                         num_layers=self.num_layers)
        self.project_0 = Conv1d(in_channels=self.rnn_0.out_channels,out_channels=1,kernel_size=1)
        self.project_1 = Conv1d(in_channels=self.rnn_1.out_channels,out_channels=1,kernel_size=1)
        self.out_channels = 2
        
    def get_config(self):
        config = {}
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
        post_rnn_0 = self.rnn_0(feature_for_gate_0,lengths)
        result_0,lengths,_ = self.project_0(post_rnn_0,lengths)
        gated_result_0 = torch.sigmoid(result_0)
        gated_x = feature_for_gate_1*gated_result_0
        post_rnn_1 = self.rnn_1(gated_x,lengths)
        result_1,lengths,_ = self.project_1(post_rnn_1,lengths)
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
        
GRU_INIT_MODE = {None:None,'bias_shift':customized_init_gru}
RNN_TYPES = {'GRU':GRU,'LSTM':LSTM,'GatedStackGRU':GatedStackGRU}
