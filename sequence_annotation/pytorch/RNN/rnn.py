import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,PackedSequence
from .customize_layer import BasicModel

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
    bidirection = torch.zeros(2*N,L,C)
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

def forward(cell, x, state):
    outputs = []
    N,L,C = x.shape
    for i in range(L):
        out, state = cell(x[:,i].squeeze(1), state)
        outputs += [out.unsqueeze(1)]
    return torch.cat(outputs,1)

class RNN(nn.Module):
    def __init__(self,rnn,init_value=0,train_init_value=True):
        super().__init__()
        self.rnn = rnn
        self.train_init_value = train_init_value
        self.init_value = init_value
        self.init_states = torch.nn.Parameter(torch.Tensor([init_value]*rnn.hidden_size),
                                              requires_grad=train_init_value)
        
    def forward(self,x,lengths,state=None):
        if state is None:
            state = self.init_states.repeat(len(x),1).cuda()
        x = reverse(x,lengths)
        x = forward(self.rnn,x,state)
        x = reverse(x,lengths)
        return x

class ReverseRNN(nn.Module):
    def __init__(self,rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self,x,lengths,state=None):
        x = reverse(x,lengths)
        x = self.rnn(x,lengths, state)
        x = reverse(x,lengths)
        return x
                    
class BidirectionalRNN(nn.Module):
    def __init__(self,rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self,x,lengths,state=None):
        x,lengths = to_bidirection(x,lengths)
        x = self.rnn(x, lengths, state)
        x,lengths = from_bidirection(x,lengths)
        return x

class ConcatRNN(BasicModel):
    def __init__(self,in_channels,hidden_size,num_layers,
                 rnn_cell_class,rnn_setting=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_setting = rnn_setting or {}
        self.rnn_cell_class = rnn_cell_class
        self.concat = Concat(dim=2)
        self.rnns = []
        for index in range(self.num_layers):
            rnn = self.rnn_cell_class(in_channels=in_channels,
                                      hidden_size=self.hidden_size,
                                      **rnn_setting)
            self.rnns.append(rnn)
            setattr(self, 'rnn_'+str(index), rnn)
            if hasattr(rnn,'out_channels'):
                out_channels = rnn.out_channels
            else:
                out_channels = rnn.hidden_size
            if rnn.bidirectional:
                out_channels *= 2
            in_channels += out_channels
            self.out_channels += out_channels
        self._build_layers()
        self.reset_parameters()
        
    def forward(self, x, lengths):
        #N,C,L
        x_ = x.transpose(1, 2)
        rnn_output = []
        for index in range(self.num_layers):            
            rnn = self.rnns[index]
            pre_x = x_
            x_ = rnn(x_,lengths)
            rnn_output.append(x_)
            x_,lengths = self.concat([pre_x,x_],lengths)
        x, lengths = self.concat(rnn_output,lengths)
        x = x.transpose(1, 2)
        distribution_output['rnn_result'] = x
        return x,lengths