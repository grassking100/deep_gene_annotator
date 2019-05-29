import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence,pack_padded_sequence,pad_packed_sequence
import torch
from torch.nn.init import constant_, normal_,uniform_

def _reverse(x,lengths):
    #Convert forward data data to reversed data with N-C-L shape to N-C-L shape
    N,C,L = x.shape
    concat_data=[]
    new_lengths = []
    for item,length in zip(x,lengths):
        reversed_core = item.transpose(0,1).flip(0)[:length].transpose(0,1)
        zeros = torch.zeros(C,L-length).cuda()
        temp = torch.cat([reversed_core,zeros],dim=1).reshape(1,C,L)
        concat_data.append(temp)
    reversed_ = torch.cat(concat_data,dim=0)
    return reversed_
def _forward(x,lengths):
    #Convert reversed data to forward data with N-C-L shape to N-C-L shape
    N,C,L = x.shape
    concat_data=[]
    for item,length in zip(x,lengths):    
        forward_core = item.transpose(0,1)[:length].transpose(0,1).flip(1)
        zeros=torch.zeros(C,L-length).cuda()
        temp = torch.cat([zeros,forward_core],dim=1).reshape(1,C,L)
        concat_data.append(temp)
    forwarded = torch.cat(concat_data,dim=0)
    return forwarded
def to_bidirection(x,lengths):
    #Convert data with N-C-L shape to 2N-C-L shape
    N,C,L = x.shape
    reversed_ = _reverse(x,lengths)
    concat_data=[]
    new_lengths = []
    for item,r_i,length in zip(x,reversed_,lengths):
        concat_data += [item.reshape(1,C,L),r_i.reshape(1,C,L)]
        new_lengths += [length,length]
    bidirection = torch.cat(concat_data,dim=0)
    return bidirection,new_lengths
def from_bidirection(x,lengths):
    #Convert data with 2N-C-L shape to N-2C-L shape
    N,C,L = x.shape
    concat_data = []
    f_x = torch.cat([x[index].reshape(1,C,L) for index in range(0,len(x),2)],dim=0)
    r_x = torch.cat([x[index+1].reshape(1,C,L) for index in range(0,len(x),2)],dim=0)
    lengths = [lengths[index] for index in range(0,len(x),2)]
    r_x = _forward(r_x,lengths)
    for index in range(len(f_x)):
        length=lengths[index]
        item=f_x[index]
        r_i=r_x[index]
        temp = torch.cat([item,r_i],1).reshape(1,2*C,L)
        concat_data.append(temp)
    bidirection = torch.cat(concat_data,dim=0)
    return bidirection, lengths

class RNN(nn.Module):
    def __init__(self,rnn_cell,go_backward=False,
                 bidirectional=True,init_value=1,train_init_value=True,state_number=None):
        super().__init__()
        self._rnn = rnn_cell
        self._bidirectional = bidirectional
        self._go_backward = go_backward
        self.init_states = torch.nn.Parameter(requires_grad=train_init_value)
        self._init_value = init_value
        self._state_number = state_number or 1
        if hasattr(self._rnn,'output_names'):
            self.output_names = self._rnn.output_names
        else:
            self.output_names = ['new_h']
        self.reset_parameters()

    def reset_parameters(self):
        self._rnn.reset_parameters()
        init_value = [self._init_value]*self._rnn.hidden_size
        self.init_states.data = torch.Tensor(init_value)

    def forward(self,x,lengths=None):
        N,L,C = x.size()
        if lengths is None:
            lengths = [L for _ in range(N)]
        if self._bidirectional:
            x,lengths = to_bidirection(x.transpose(1,2),lengths)
            x = x.transpose(1,2)
        elif not self._go_backward:
            pass
        else:
            x = _reverse(x.transpose(1,2),lengths).transpose(1,2)
        all_states = []
        x,batch_sizes = self._preprocess(x,batch_first,lengths)
        outputs = self._forward(x,batch_sizes)
        for output in outputs:
            #output = PackedSequence(output, batch_sizes)
            output,_ = pad_packed_sequence(output, batch_first=True)
            if self._bidirectional:
                output,_ = from_bidirection(output.transpose(1,2),lengths)
                output = output.transpose(1,2)
            elif not self._go_backward:
                pass
            else:
                output = _forward(output.transpose(1,2),lengths).transpose(1,2)
            all_states.append(output)
        return all_states

    def _preprocess(self,x,lengths):
        N,L,C = x.size()
        lengths = torch.LongTensor(lengths)
        if lengths is not None:
            x,batch_sizes = pack_padded_sequence(x,lengths,batch_first=True)
        else:
            batch_sizes = [N for _ in range(L)]
        return x,batch_sizes

    def _forward(self,x,batch_sizes):
        previous_h = self.init_states.repeat(len(x),self._state_number)
        all_states = [[] for _ in self.output_names]
        count = 0 
        for batch_size in batch_sizes:
            outputs = self._rnn(x[count:count+batch_size],(previous_h[:batch_size]))
            if not isinstance(outputs,tuple):
                outputs = [outputs]
            previous_h= outputs[0]
            count+=batch_size
            for index,o in enumerate(outputs):
                all_states[index].append(o)
        for index in range(len(self.output_names)):
            all_states[index] = torch.cat(all_states[index], 0)
        return all_states