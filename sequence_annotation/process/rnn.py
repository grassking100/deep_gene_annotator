from abc import abstractmethod
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
from .customized_layer import BasicModel,Concat
from .cnn import Conv1d
from .utils import xavier_uniform_extend_
from torch.nn.init import zeros_,constant_,orthogonal_

from torch.nn.init import _calculate_fan_in_and_fan_out

def xav_gru_init(rnn,mode=None):
    mode = mode or 'fan_in'
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            w_ir, w_iz, w_in = getattr(rnn,'weight_ih_l{}{}'.format(index,suffix)).chunk(3, 0)
            w_hr, w_hz, w_hn = getattr(rnn,'weight_hh_l{}{}'.format(index,suffix)).chunk(3, 0)
            for var in [w_ir, w_iz, w_in, w_hz, w_hr,w_hn]:
                xavier_uniform_extend_(var,mode=mode)

def orth_xav_gru_init(rnn,mode=None):
    mode = mode or 'fan_in'
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            w_ir, w_iz, w_in = getattr(rnn,'weight_ih_l{}{}'.format(index,suffix)).chunk(3, 0)
            w_hr, w_hz, w_hn = getattr(rnn,'weight_hh_l{}{}'.format(index,suffix)).chunk(3, 0)
            orthogonal_(w_hn)
            for var in [w_ir, w_iz, w_in, w_hz, w_hr]:
                xavier_uniform_extend_(var,mode=mode)
                
def bias_zero_gru_init(rnn):
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            ri_bias,zi_bias,ni_bias = getattr(rnn,'bias_ih_l{}{}'.format(index,suffix)).chunk(3, 0)
            rh_bias,zh_bias,nh_bias = getattr(rnn,'bias_hh_l{}{}'.format(index,suffix)).chunk(3, 0)
            for var in [ri_bias,zi_bias,ni_bias,rh_bias,zh_bias,nh_bias]:
                zeros_(var)
                
def bias_xav_gru_init(rnn,mode=None):
    mode = mode or 'fan_in'
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            ri_bias,zi_bias,ni_bias = getattr(rnn,'bias_ih_l{}{}'.format(index,suffix)).chunk(3, 0)
            rh_bias,zh_bias,nh_bias = getattr(rnn,'bias_hh_l{}{}'.format(index,suffix)).chunk(3, 0)
            w_ir = getattr(rnn,'weight_ih_l{}{}'.format(index,suffix)).chunk(3, 0)[0]
            w_hr = getattr(rnn,'weight_hh_l{}{}'.format(index,suffix)).chunk(3, 0)[0]
            fan_in, fan_out = _calculate_fan_in_and_fan_out(w_ir)
            h_n = fan_out
            if mode == 'both':
                n = float(fan_in + fan_out)/2
            elif mode == 'fan_in':
                n = fan_in
            else:
                n = fan_out

            for var in [rh_bias,zh_bias,nh_bias]:
                xavier_uniform_extend_(var,n=h_n)

            for var in [ri_bias,zi_bias,ni_bias]:
                xavier_uniform_extend_(var,n=n)
                
def xav_bias_zero_gru_init(rnn,mode=None):
    xav_gru_init(rnn,mode)
    bias_zero_gru_init(rnn)
    
def xav_bias_xav_gru_init(rnn,mode=None):
    xav_gru_init(rnn,mode)
    bias_xav_gru_init(rnn,mode)

def orth_xav_bias_zero_gru_init(rnn,mode=None):
    orth_xav_gru_init(rnn,mode=mode)
    bias_zero_gru_init(rnn)

def orth_both_xav_bias_zero_gru_init(rnn):
    return orth_xav_bias_zero_gru_init(rnn,'both')

def orth_in_xav_bias_zero_gru_init(rnn):
    return orth_xav_bias_zero_gru_init(rnn,'fan_in')

def orth_out_xav_bias_zero_gru_init(rnn):
    return orth_xav_bias_zero_gru_init(rnn,'fan_out')

def both_xav_bias_zero_gru_init(rnn):
    return xav_bias_zero_gru_init(rnn,'both')

def in_xav_bias_zero_gru_init(rnn):
    return xav_bias_zero_gru_init(rnn,'fan_in')

def out_xav_bias_zero_gru_init(rnn):
    return xav_bias_zero_gru_init(rnn,'fan_out')

def in_xav_bias_xav_gru_init(rnn):
    return xav_bias_xav_gru_init(rnn,'fan_in')

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

GRU_INIT_MODE = {
    None:None,
    'bias_shift':customized_init_gru,
    'orth_xav_bias_zero_gru_init':orth_xav_bias_zero_gru_init,
    'orth_both_xav_bias_zero_gru_init':orth_both_xav_bias_zero_gru_init,
    'orth_in_xav_bias_zero_gru_init':orth_in_xav_bias_zero_gru_init,
    'orth_out_xav_bias_zero_gru_init':orth_out_xav_bias_zero_gru_init,
    'both_xav_bias_zero_gru_init':both_xav_bias_zero_gru_init,
    'in_xav_bias_zero_gru_init':in_xav_bias_zero_gru_init,
    'out_xav_bias_zero_gru_init':out_xav_bias_zero_gru_init,
    'in_xav_bias_xav_gru_init':in_xav_bias_xav_gru_init
}
            
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
    def __init__(self,in_channels,train_init_value=False,init_value=None,customized_init=None,**kwargs):
        super().__init__()
        if isinstance(customized_init,str):
            customized_init = GRU_INIT_MODE[customized_init]
        self.customized_init = customized_init
        self.rnn = self._create_rnn(in_channels,**kwargs)
        self.kwargs = kwargs
        self.in_channels = in_channels
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
    def _create_rnn(self,in_channels,**kwargs):
        pass
    
    def get_config(self):
        config = super().get_config()
        config['train_init_value'] = self.train_init_value
        config['init_value'] = self.init_value
        config['setting'] = self.kwargs
        config['customized_init'] = str(self.customized_init)
        return config
    
    def reset_parameters(self):
        super().reset_parameters()
        if self.customized_init is not None:
            self.customized_init(self.rnn)
    
class GRU(_RNN):
    def _create_rnn(self,in_channels,**kwargs):
        return nn.GRU(in_channels,**kwargs)
    
    def forward(self,x,lengths=None,state=None):
        if lengths is None:
            lengths = [x.shape[2]]*len(x)
        if state is None:
            state = self.init_states.unsqueeze(1).repeat(1,len(x),1)
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x
    
class LSTM(_RNN):
    def _create_rnn(self,in_channels,**kwargs):
        return nn.LSTM(in_channels,**kwargs)
    
    def forward(self,x,lengths,state=None):
        if state is None:
            state = self.init_states.repeat(len(x),1).cuda()
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x

class ProjectedGRU(BasicModel):
    def __init__(self,in_channels,out_channels,hidden_size=None,
                 num_layers=None,bias=True,batch_first=False,dropout=None,
                 bidirectional=False,
                 customized_cnn_init=None,customized_gru_init=None,
                 norm_type=None,norm_mode=None,name=None,**kwargs):
        super().__init__()
        self.hidden_size = hidden_size or 16
        self.num_layers = num_layers or 1
        self.bias = bias
        self.batch_first = batch_first 
        self.dropout = dropout or 0
        self.bidirectional = bidirectional
        if name is None:
            self.name = ''
        else:
            self.name = "{}_".format(name)
        #print(kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rnn = GRU(in_channels=self.in_channels,hidden_size=self.hidden_size,
                       num_layers=self.num_layers,bias=self.bias,
                       batch_first=self.batch_first,dropout=self.dropout,
                       bidirectional=self.bidirectional,
                       customized_init=customized_gru_init)
        self.norm_type = norm_type
        if self.norm_type is not None:
            self.norm = self.norm_type(in_channel[self.norm_mode])
        self.project = Conv1d(in_channels=self.rnn.out_channels,
                              out_channels=self.out_channels,kernel_size=1,
                              customized_init=customized_cnn_init)
        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config['rnn'] = self.rnn.get_config()
        config['project'] = self.project.get_config()
        config['norm_type'] = self.norm_type
        config['name'] = self.name        
        config['hidden_size'] = self.hidden_size
        config['num_layers'] = self.num_layers
        config['bias'] = self.bias
        config['batch_first'] = self.batch_first
        config['dropout'] = self.dropout
        config['bidirectional'] = self.bidirectional
        
        return config
        
    def forward(self,x,lengths,return_intermediate=False):
        post_rnn = self.rnn(x,lengths)
        self.update_distribution(post_rnn,key='{}rnn'.format(self.name))
        if self.norm_type is not None:
            post_rnn = self.norm(post_rnn,lengths)
        result,lengths,_,_ = self.project(post_rnn,lengths=lengths)
        if return_intermediate:
            return result,post_rnn
        else:
            return result

RNN_TYPES = {'GRU':GRU,'LSTM':LSTM,'ProjectedGRU':ProjectedGRU}
