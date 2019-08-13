from abc import abstractmethod
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
from .customized_layer import BasicModel,Concat

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
    def __init__(self,in_channels,train_init_value=False,init_value=None,**rnn_setting):
        super().__init__()
        self.rnn = self._create_rnn(in_channels,**rnn_setting)
        self.rnn_setting = rnn_setting
        self.in_channels = in_channels
        self.out_channels = self.rnn.hidden_size
        self.train_init_value = train_init_value
        self.init_value = init_value or 0
        direction = 2 if self.rnn.bidirectional else 1
        val = [self.init_value]*self.rnn.hidden_size
        val = torch.Tensor(val).unsqueeze(0).repeat(self.rnn.num_layers*direction,1)
        self.init_states = torch.nn.Parameter(val,requires_grad=train_init_value)
        self.batch_first = self.rnn.batch_first
        if self.rnn.bidirectional:
            self.out_channels *= 2

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
        return config
        
class GRU(_RNN):
    def _create_rnn(self,in_channels,**rnn_setting):
        return nn.GRU(in_channels,**rnn_setting)
    
    def forward(self,x,lengths,state=None):
        if state is None:
            state = self.init_states.unsqueeze(1).repeat(1,len(x),1)
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x

class LSTM(_RNN):
    def _create_rnn(self,in_channels,**rnn_setting):
        return nn.LSTM(in_channels,**rnn_setting)
    
    def forward(self,x,lengths,state=None):
        if state is None:
            state = self.init_states.repeat(len(x),1).cuda()
        x = _forward(self.rnn,x,lengths,state,self.batch_first)
        return x

RNN_TYPES = {'GRU':GRU,'LSTM':LSTM}
    
def reverse(x,lengths):
    #Convert forward data data to reversed data with N-L-C shape to N-L-C shape
    N,L,C = x.shape
    concat_data=[]
    reversed_ = torch.zeros(*x.shape).cuda()
    for index,(item,length) in enumerate(zip(x,lengths)):
        reversed_core = item[:length].flip(0)
        reversed_[index,:length] = reversed_core
    return reversed_

def to_bidirection(x,lengths):
    #Convert data with N-L-C shape to 2N-L-C shape
    N,L,C = x.shape
    bidirection = torch.zeros(2*N,L,C).cuda()
    reversed_ = reverse(x,lengths)
    bidirection[:N] = x
    bidirection[N:] = reversed_
    new_lengths = []
    for length in lengths:
        new_lengths += [length]*2
    return bidirection,new_lengths

def from_bidirection(x,lengths):
    #Convert data with 2N-L-C shape to N-L-2C shape
    N,C,L = x.shape
    half_N = int(N/2)
    forward = x[:half_N]
    reverse = x[half_N:]
    bidirection = torch.cat([forward,reverse],dim=2)
    new_lengths = [lengths[index] for index in range(0,len(x),2)]
    return bidirection, new_lengths

class ReverseRNN(BasicModel):
    def __init__(self,rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self,x,lengths,state=None):
        x = reverse(x,lengths)
        x = self.rnn(x,lengths, state)
        x = reverse(x,lengths)
        return x
                
class BidirectionalRNN(BasicModel):
    def __init__(self,rnn):
        super().__init__()
        self.rnn = rnn
        self.out_channels = self.rnn.out_channels * 2

    def forward(self,x,lengths,state=None):
        x,lengths = to_bidirection(x,lengths)
        x = self.rnn(x, lengths, state)
        x,lengths = from_bidirection(x,lengths)
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
        config['rnn_setting'] = self.rnn_setting
        config['rnn_type'] = self.rnn_type
        return config
