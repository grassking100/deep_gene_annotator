import torch
from .customized_layer import BasicModel
from .rnn import RNN_TYPES, _RNN

def _reverse(x,lengths):
    #Convert forward data data to reversed data with N-C-L shape to N-C-L shape
    x = x.transpose(1,2)
    N,L,C = x.shape
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
RNN_TYPES.update({'ConcatGRU':ConcatGRU})
