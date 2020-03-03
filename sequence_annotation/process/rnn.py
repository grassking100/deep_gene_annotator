from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .customized_layer import BasicModel
from .cnn import Conv1d
from .utils import xavier_uniform_extend_
from torch.nn.init import zeros_

def separate_xav_rnn_init(rnn,mode=None,chunk_size=None):
    chunk_size = chunk_size or 3
    mode = mode or 'fan_in'
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            weights = []
            weight_ih = getattr(rnn,'weight_ih_l{}{}'.format(index,suffix)).chunk(chunk_size, 0)
            weight_hh = getattr(rnn,'weight_hh_l{}{}'.format(index,suffix)).chunk(chunk_size, 0)
            weights += weight_ih
            weights += weight_hh
            for var in weights:
                xavier_uniform_extend_(var,mode=mode)
                
def xav_rnn_init(rnn,mode=None,chunk_size=None):
    chunk_size = chunk_size or 3
    mode = mode or 'fan_in'
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            weights = []
            weight_ih = getattr(rnn,'weight_ih_l{}{}'.format(index,suffix)).chunk(chunk_size, 0)
            weight_hh = getattr(rnn,'weight_hh_l{}{}'.format(index,suffix)).chunk(chunk_size, 0)
            weights += weight_ih
            weights += weight_hh
            i_o,i_i = weight_ih[0].shape
            h_o,h_i = weight_hh[0].shape
            if i_o != h_o:
                raise Exception()
            if mode =='fan_in':
                n = i_i + h_i
            elif mode =='fan_out':
                n = i_o
            elif mode =='both':
                n = (i_o+i_i + h_i)/2
            else:
                raise Exception()
            for var in weights:
                xavier_uniform_extend_(var,n=n)
                
def bias_zero_rnn_init(rnn,chunk_size=None):
    chunk_size = chunk_size or 3
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            weights = []
            weights += getattr(rnn,'bias_ih_l{}{}'.format(index,suffix)).chunk(chunk_size, 0)
            weights += getattr(rnn,'bias_hh_l{}{}'.format(index,suffix)).chunk(chunk_size, 0)
            for var in weights:
                zeros_(var)
                
def xav_bias_zero_rnn_init(rnn,mode=None,chunk_size=None):
    xav_rnn_init(rnn,mode,chunk_size)
    bias_zero_rnn_init(rnn,chunk_size)
    
def separate_xav_bias_zero_rnn_init(rnn,mode=None,chunk_size=None):
    separate_xav_rnn_init(rnn,mode,chunk_size)
    bias_zero_rnn_init(rnn,chunk_size)

def in_xav_bias_zero_gru_init(rnn):
    return xav_bias_zero_rnn_init(rnn,'fan_in')

def out_xav_bias_zero_gru_init(rnn):
    return xav_bias_zero_rnn_init(rnn,'fan_out')

def separate_in_xav_bias_zero_gru_init(rnn):
    return separate_xav_bias_zero_rnn_init(rnn,'fan_in')

def separate_out_xav_bias_zero_gru_init(rnn):
    return separate_xav_bias_zero_rnn_init(rnn,'fan_out')

RNN_INIT_MODE = {
    None:None,
    'in_xav_bias_zero_gru_init':in_xav_bias_zero_gru_init,
    'out_xav_bias_zero_gru_init':out_xav_bias_zero_gru_init,
    'separate_in_xav_bias_zero_gru_init':separate_in_xav_bias_zero_gru_init,
    'separate_out_xav_bias_zero_gru_init':separate_out_xav_bias_zero_gru_init,
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
    def __init__(self,in_channels,customized_init=None,**kwargs):
        super().__init__()
        if isinstance(customized_init,str):
            customized_init = RNN_INIT_MODE[customized_init]
        self.customized_init = customized_init
        self.rnn = self._create_rnn(in_channels,**kwargs)
        self.kwargs = kwargs
        self.in_channels = in_channels
        self.out_channels = self.rnn.hidden_size
        if self.rnn.bidirectional:
            self.out_channels *= 2
        self.batch_first = self.rnn.batch_first if hasattr(self.rnn,'batch_first') else True
        self.reset_parameters()

    @property
    def dropout(self):
        return self.rnn.dropout
        
    @abstractmethod
    def _create_rnn(self,in_channels,**kwargs):
        pass
    
    def get_config(self):
        config = super().get_config()
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
    
    def forward(self,x,lengths,state=None,**kwargs):
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x
    
class LSTM(_RNN):
    def _create_rnn(self,in_channels,**kwargs):
        return nn.LSTM(in_channels,**kwargs)
    
    def forward(self,x,lengths,state=None,**kwargs):
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x

class ProjectedRNN(BasicModel):
    def __init__(self,in_channels,out_channels,
                 customized_cnn_init=None,customized_rnn_init=None,
                 name=None,output_act=None,is_gru=True,**kwargs):
        super().__init__()
        self.output_act = output_act
        if name is None or name == '':
            self.name = 'rnn'
        else:
            self.name = "{}_rnn".format(name)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if is_gru:
            rnn_class = GRU
        else:
            rnn_class = LSTM
        self.rnn = rnn_class(in_channels=self.in_channels,
                             customized_init=customized_rnn_init,**kwargs)
        self.project = Conv1d(in_channels=self.rnn.out_channels,
                              out_channels=self.out_channels,kernel_size=1,
                              customized_init=customized_cnn_init)
        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config['rnn'] = self.rnn.get_config()
        config['project'] = self.project.get_config()
        config['name'] = self.name
        config['output_act'] = self.output_act
        return config
        
    def forward(self,x,lengths,return_intermediate=False,**kwargs):
        post_rnn = self.rnn(x,lengths)
        self.update_distribution(post_rnn,key=self.name)
        if self.rnn.dropout > 0 and self.training:
            post_rnn = F.dropout(post_rnn,self.rnn.dropout,self.training)
        result,lengths,_,_ = self.project(post_rnn,lengths=lengths)

        if self.output_act is not None:
            if self.output_act == 'softmax':
                result = torch.softmax(result,1)
            elif self.output_act == 'sigmoid':
                result = torch.sigmoid(result)
            else:
                raise Exception("Wrong activation name, {}".format(self.output_act))
        
        if return_intermediate:
            return result,post_rnn
        else:
            return result

RNN_TYPES = {'GRU':GRU,'LSTM':LSTM,'ProjectedRNN':ProjectedRNN}
