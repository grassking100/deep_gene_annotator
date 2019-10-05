from abc import abstractmethod
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
from torch.nn.init import xavier_uniform_,zeros_,orthogonal_, _calculate_fan_in_and_fan_out, _no_grad_uniform_,constant_
from .customized_layer import BasicModel,Concat, Conv1d

def xavier_uniform_in_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1 / float(fan_in))
    bound = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -bound, bound)
        
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

GRU_INIT_MODE = {None:None,'bias_shift':customized_init_gru}
        
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
        config['out_channels'] = self.hidden_size
        config['num_layers'] = self.num_layers
        return config
        
    def forward(self,x,lengths):
        post_rnn_0 = self.rnn_0(x,lengths)
        self._distribution['post_rnn_0'] = post_rnn_0
        result_0,lengths,_ = self.project_0(post_rnn_0,lengths)
        self._distribution['result_0'] = result_0
        gated_result_0 = torch.sigmoid(result_0)
        self._distribution['gated_result_0'] = gated_result_0
        #print(gated_result_0.shape)
        #print(x.shape)
        gated_x = x*gated_result_0
        #print(gated_x.shape)
        self._distribution['gated_x'] = gated_x
        post_rnn_1 = self.rnn_1(gated_x,lengths)
        self._distribution['post_rnn_1'] = post_rnn_1
        result_1,lengths,_ = self.project_1(post_rnn_1,lengths)
        self._distribution['result_1'] = result_1
        gated_result_1 = torch.sigmoid(result_1)
        self._distribution['gated_result_1'] = gated_result_1
        result = torch.cat([gated_result_0,gated_result_1],1)
        self._distribution['gated_stack_result'] = result
        return result
        
RNN_TYPES = {'GRU':GRU,'LSTM':LSTM,'GatedStackGRU':GatedStackGRU}
        
def _reverse(x,lengths):
    #Convert forward data data to reversed data with N-C-L shape to N-C-L shape
    x = x.transpose(1,2)
    N,L,C = x.shape
    concat_data=[]
    reversed_ = torch.zeros(*x.shape).cuda()
    for index,(item,length) in enumerate(zip(x,lengths)):
        reversed_core = item[:length].flip(0)
        reversed_[index,:length] = reversed_core
    reversed_ = reversed_.transpose(1,2)    
    return reversed_

def _to_bidirection(x,lengths):
    #Convert data with N-C-L shape to two tensors with N-C-L shape
    N,L,C = x.shape
    reversed_ = _reverse(x,lengths)
    return x,reversed_

def _from_bidirection(forward,reversed_,lengths):
    #Convert two tensors with N-C-L shape to one tensors with N-2C-L shape
    reversed_ = _reverse(reversed_,lengths)
    bidirection = torch.cat([forward,reversed_],dim=1)
    return bidirection

class ReverseRNN(BasicModel):
    def __init__(self,rnn):
        super().__init__()
        self.rnn = rnn
        self.out_channels = self.rnn.out_channels
        self.hidden_size = self.rnn.hidden_size
        
    def forward(self,x,lengths,state=None):
        x = _reverse(x,lengths)
        x = self.rnn(x,lengths, state)
        x = _reverse(x,lengths)
        return x
                
class BidirectionalRNN(BasicModel):
    def __init__(self,forward_rnn,backward_rnn):
        super().__init__()
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn
        if self.forward_rnn.out_channels != self.backward_rnn.out_channels:
            raise Exception("Forward and backward RNNs' out channels should be the same")
        self.out_channels = self.forward_rnn.out_channels
        self.hidden_size = self.forward_rnn.hidden_size
     
    @property
    def bidirectional(self):
        return True

    def forward(self,x,lengths,forward_state=None,reverse_state=None):
        forward_x,reversed_x = _to_bidirection(x,lengths)
        forward_x = self.forward_rnn(forward_x, lengths, forward_state)
        reversed_x = self.backward_rnn(reversed_x, lengths, reverse_state)
        x = _from_bidirection(forward_x,reversed_x,lengths)
        return x

class ConcatRNN(BasicModel):
    def __init__(self,in_channels,hidden_size,num_layers,
                 rnn_type,**rnn_setting):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_setting = rnn_setting
        if isinstance(rnn_type,str):
            try:
                self.rnn_type = RNN_TYPES[rnn_type]
            except:
                raise Exception("{} is not supported".format(rnn_type))
        else:        
            self.rnn_type = rnn_type                
        self.concat = Concat(dim=1)
        self.rnns = []
        
        for index in range(self.num_layers):
            rnn = self.rnn_type(in_channels=in_channels,
                                hidden_size=self.hidden_size,
                                **rnn_setting)
            self.rnns.append(rnn)
            out_channels = rnn.out_channels
            setattr(self, 'rnn_'+str(index), rnn)
            in_channels += out_channels
        self.out_channels = out_channels*self.num_layers
        self.reset_parameters()
        
    def forward(self, x, lengths,state=None):
        #N,C,L
        rnn_output = []
        for index in range(self.num_layers):            
            rnn = self.rnns[index]
            pre_x = x
            x = rnn(x,lengths,state)
            rnn_output.append(x)
            x,lengths = self.concat([pre_x,x],lengths)
        x, lengths = self.concat(rnn_output,lengths)
        self._distribution['rnn_result'] = x
        return x
    
    def get_config(self):
        config = {}
        config['in_channels'] = self.in_channels
        config['hidden_size'] = self.hidden_size
        config['num_layers'] = self.num_layers
        rnn_setting = dict(self.rnn_setting)
        rnn_setting['customized_gru_init'] = str(rnn_setting['customized_gru_init'])
        config['rnn_setting'] = rnn_setting
        config['rnn_type'] = self.rnn_type
        return config
