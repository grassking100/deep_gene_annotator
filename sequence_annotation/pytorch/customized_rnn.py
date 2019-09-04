import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_,uniform_
from .customized_layer import BasicModel
from .noisy_activation import SymNoisyHardSigmoid
from .rnn import _RNN,ConcatRNN,BidirectionalRNN,LSTM,GRU

def to_bidirectional(rnn_class,**rnn_setting):
    forward_rnn = rnn_class(**rnn_setting)
    backward_rnn = rnn_class(**rnn_setting)
    rnn = BidirectionalRNN(forward_rnn,backward_rnn)
    return rnn

class RNN(_RNN):
    def forward(self,x,lengths,state=None):
        #Input:N,C,L, Output: N,C,L
        if not self.batch_first:
            x = x.transpose(0,1)
        x = x.transpose(1,2)
        if state is None:
            state = self.init_states.repeat(len(x),1).cuda()    
        outputs = []
        N,L,C = x.shape
        for i in range(L):
            out, state = self.rnn(x[:,i], state)
            outputs += [out.unsqueeze(1)]
        x = torch.cat(outputs,1)
        x = x.transpose(1,2)
        if not self.batch_first:
            x = x.transpose(0,1)
        return x

    def get_config(self):
        config = super().get_config()
        config['cell'] = str(self.cell)
        return config
    
class StackRNN(BasicModel):
    def __init__(self,rnn_class,num_layers=1,**rnn_setting):
        super().__init__()
        self.bidirectional = False
        rnn_setting = dict(rnn_setting)
        if 'bidirectional' in rnn_setting:
            self.bidirectional = rnn_setting['bidirectional']
            del rnn_setting['bidirectional']
        self.num_layers = num_layers
        self.rnns = []
        for index in range(self.num_layers):
            if self.bidirectional:
                rnn=to_bidirectional(rnn_class,**rnn_setting)
            else:
                rnn=rnn_class(**rnn_setting)
            dir_num = 2 if self.bidirectional else 1
            rnn_setting['in_channels'] = rnn.out_channels*dir_num
            setattr(self,'rnn_{}'.format(index),rnn)
            self.rnns.append(rnn)
        self.out_channels = rnn_setting['in_channels']

    def forward(self,x,lengths):
        for index in range(self.num_layers):
            rnn = self.rnns[index]
            x = rnn(x,lengths)
        return x

    def get_config(self):
        config = {}
        for index in range(self.num_layers):
            config = self.rnns[index].get_config()
            config['rnn_{}'.format(index)] = config
        return config

class SuperMinimalRNNCell(BasicModel):
    def __init__(self, in_channels, hidden_size,batch_first=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        if not batch_first:
            raise Exception("{} should set batch_first to True".format(self.__class__.__name__))
        self.gate_num=1
        self.gate_weights = nn.Parameter(torch.empty(hidden_size*self.gate_num, in_channels+hidden_size))
        self.weights = nn.Parameter(torch.empty(hidden_size*self.gate_num, in_channels))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.gate_bias = nn.Parameter(torch.empty(hidden_size*self.gate_num))
        
        self.act = SymNoisyHardSigmoid(c=1e-3)
        self.reset_parameters()

    def reset_parameters(self):
        gate_bound = (2/(self.hidden_size*(self.gate_num+1)+self.in_channels))**0.5
        input_bound = (2/(self.in_channels+self.hidden_size))**0.5
        uniform_(self.gate_weights,-gate_bound,gate_bound)
        constant_(self.gate_bias,0.5)
        uniform_(self.weights,-input_bound,input_bound)
        #bias_bound = (2/(self.hidden_size))**0.5
        uniform_(self.bias,0,0)

    def forward(self, x, state):
        #input shape should be (number,feature size)
        concated = torch.cat([x,state],1)
        pre_gate = F.linear(concated, self.gate_weights, self.gate_bias)
        gate = self.act(pre_gate)
        #print(pre_gate,gate)
        gate_f = torch.chunk(gate, self.gate_num, dim=1)[0]
        values = torch.tanh(F.linear(x, self.weights, self.bias))
        #values = self.bias)
        new_state = state*gate_f+ values*(1-gate_f)
        return new_state,new_state
    
class _SuperMinimalRNN(RNN):
    def _create_rnn(self,in_channels,**rnn_setting):
        return SuperMinimalRNNCell(in_channels,**rnn_setting)

class SuperMinimalRNN(BasicModel):
    def __init__(self,**rnn_setting):
        super().__init__()
        self.rnn = StackRNN(_SuperMinimalRNN,**rnn_setting)
        self.out_channels = self.rnn.out_channels

    def forward(self,x,lengths):
        return self.rnn(x,lengths)

    def get_config(self):
        config = self.rnn.get_config()
        return config

class ConcatGRU(BasicModel):
    def __init__(self,**rnn_setting):
        super().__init__()
        self.rnn = ConcatRNN(rnn_type='GRU',**rnn_setting)
        self.out_channels = self.rnn.out_channels

    def forward(self,x,lengths):
        return self.rnn(x, lengths)
    
    def get_config(self):
        config = {'rnn':self.rnn.get_config(),
                  'in_channels':self.rnn.in_channels,
                  'out_channels':self.out_channels,
                 }
        return config
    
RNN_TYPES = {'GRU':GRU,'LSTM':LSTM,'ConcatGRU':ConcatGRU,'SuperMinimalRNN':SuperMinimalRNN}
