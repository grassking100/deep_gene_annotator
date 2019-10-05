import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import constant_,uniform_
from .customized_layer import BasicModel,Conv1d
from .noisy_activation import SymNoisyHardSigmoid
from .rnn import _RNN,ConcatRNN,BidirectionalRNN,LSTM,GRU,RNN_TYPES

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

class _MinimalInyRNNCell(BasicModel):
    def __init__(self, in_channels, hidden_size,batch_first=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        if not batch_first:
            raise Exception("{} should set batch_first to True".format(self.__class__.__name__))
        self.cnn = Conv1d(in_channels=in_channels,out_channels=hidden_size,kernel_size=1)
        self.act = nn.Hardtanh(min_val=0)
        self.reset_parameters()

    def forward(self, x, state,precompute=False):
        #input shape should be (number,feature size)
        if precompute:
            x = self.cnn(x.transpose(1,2))[0].transpose(1,2)
            return x
        else:
            x = self.act(x + state)
        return x,x
    
class _MinimalInyRNN(_RNN):
    def _create_rnn(self,in_channels,**rnn_setting):
        return _MinimalInyRNNCell(in_channels,**rnn_setting)
    def forward(self,x,lengths,state=None):
        #Input:N,C,L, Output: N,C,L
        if not self.batch_first:
            x = x.transpose(0,1)
        x = x.transpose(1,2)
        if state is None:
            state = self.init_states.repeat(len(x),1).cuda()    
        outputs = []
        N,L,C = x.shape
        x = self.rnn(x, None,precompute=True)
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

class MinimalInyRNN(BasicModel):
    def __init__(self,**rnn_setting):
        super().__init__()
        self.rnn = StackRNN(_MinimalInyRNN,**rnn_setting)
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
    
RNN_TYPES = dict(RNN_TYPES)
RNN_TYPES.update({'ConcatGRU':ConcatGRU,'MinimalInyRNN':MinimalInyRNN})
