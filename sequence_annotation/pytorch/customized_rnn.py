import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_,uniform_
from .customized_layer import BasicModel
from .rnn import _RNN,ConcatRNN,BidirectionalRNN,LSTM,GRU

class RNN(_RNN):
    def __init__(self,cell,**rnn_setting):
        self.cell = cell
        super().__init__(**rnn_setting)

    def _create_rnn(self,in_channels,**rnn_setting):
        return self.cell(in_channels,rnn_setting)

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
            out, state = cell(x[:,i].squeeze(1), state)
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

class IndyGRUCell(BasicModel):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.gate_weights = nn.Parameter(torch.empty(hidden_size*2, in_channels+hidden_size))
        self.weights = nn.Parameter(torch.empty(hidden_size, in_channels))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        self.gate_bias = nn.Parameter(torch.empty(hidden_size*2))
        self.reset_parameters()

    def reset_parameters(self):
        gate_bound = (2/(self.hidden_size*3+self.in_channels))**0.5
        input_bound = (2/(self.in_channels+self.hidden_size))**0.5
        uniform_(self.gate_weights,-gate_bound,gate_bound)
        uniform_(self.weights,-input_bound,input_bound)
        constant_(self.gate_bias,0.5)
        constant_(self.bias,0)

    def forward(self, x, state):
        #input shape should be (number,feature size)
        concated = torch.cat([x,state],1)
        values = torch.tanh(F.linear(x, self.weights, self.bias))
        pre_gate = F.linear(concated, self.gate_weights, self.gate_bias)
        gate = torch.sigmoid(pre_gate)
        gate_f, gate_i = torch.chunk(gate, 2, dim=1)
        new_state = state*gate_f+ values*gate_i
        return new_state,new_state
    
class IndyGRU(BasicModel):
    def __init__(self,in_channels, hidden_size,**rnn_setting):
        super().__init__()
        self.out_channels = hidden_size
        cell = IndyGRUCell(in_channels, hidden_size)
        self.rnn = RNN(cell,**rnn_setting)

    def forward(self,x,lengths,state=None):
        return self.rnn(x,lengths,state)
    
class ConcatGRU(BasicModel):
    def __init__(self,in_channels,hidden_size,num_layers,**rnn_setting):
        super().__init__()
        rnn = ConcatRNN(in_channels,hidden_size,num_layers,rnn_type='GRU',**rnn_setting)
        self.out_channels = rnn.out_channels
        self.rnn = rnn

    def forward(self,x,lengths):
        return self.rnn(x, lengths)
    
    def get_config(self):
        config = {'rnn':self.rnn.get_config(),
                  'in_channels':self.rnn.in_channels,
                  'out_channels':self.out_channels,
                 }
        return config
    
RNN_TYPES = {'GRU':GRU,'LSTM':LSTM,'ConcatGRU':ConcatGRU}
