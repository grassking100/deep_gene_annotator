from abc import ABCMeta
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F

PADDING_HANDLE = ['valid','same']

class Conv1d(nn.Conv1d):
    def __init__(self,padding_handle=None,*args,**kwargs):
        super().__init__(*args,**kwargs)
        padding_handle = padding_handle or 'valid'
        if padding_handle in PADDING_HANDLE:
            self._padding_handle = padding_handle
        else:
            raise Exception("Invalid mode {} to handle padding".format(padding_handle))
        self._pad_func = None
        if self._padding_handle != 'valid':
            if self.padding != (0,) or self.dilation != (1,) or self.stride != (1,):
                raise Exception("The padding_handle sholud be valid to set padding, dilation and stride at the same time")
            kernek_size = self.kernel_size[0]
            if self._padding_handle == 'same' and kernek_size>1:
                if kernek_size%2 == 0:
                    bound = int(kernek_size/2)
                    self._pad_func = nn.ConstantPad1d((bound-1,bound),0)
                else:
                    bound = int((kernek_size-1)/2)
                    self._pad_func = nn.ConstantPad1d((bound,bound),0)

    def forward(self,x,lengths=None):
        #N,C,L
        if self._pad_func is not None:
            x = self._pad_func(x)
        x = super().forward(x)
        if lengths is not None:
            if self._padding_handle == 'same':
                new_lengths = lengths
            else:
                padding = sum(self.padding)
                dilation= sum(self.dilation)
                stride= sum(self.stride)
                change = 2*padding-dilation*(self.kernel_size[0]-1)-1
                new_lengths = [math.floor((length + change)/stride) + 1 for length in lengths]
                x = x[:,:,:max(new_lengths)]
            return x, new_lengths
        else:
            return x

class Concat(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def forward(self,x,lengths=None):
        #N,C,L
        min_length = min(item.shape[2] for item in x)
        data_list = [item[:,:,:min_length] for item in x]
        new_x = torch.cat(data_list,*self.args,**self.kwargs)
        if lengths is not None:
            return new_x, lengths
        else:
            return new_x

def add(lhs,rhs,lengths=None):
    #N,C,L
    min_length = min(item.shape[2] for item in [lhs,rhs])
    lhs = lhs[:,:,:min_length]
    rhs = rhs[:,:,:min_length]
    new_x = lhs + rhs
    if lengths is not None:
        return new_x, lengths
    else:
        return new_x

class PWM(nn.Module):
    def __init__(self,epsilon=None):
        super().__init__()
        if epsilon is None:
            self._epsilon = 1e-32
        else:
            self._epsilon = epsilon

    def forward(self,x):
        #N,C,L
        channel_size = x.shape[1]
        freq = F.softmax(x,dim=1)
        freq_with_background = freq * channel_size
        inform = (freq*(freq_with_background+self._epsilon).log2_()).sum(1)
        if len(x.shape)==3:
            return (freq.transpose(0,1)*inform).transpose(0,1)
        elif len(x.shape)==2:
            return (freq.transpose(0,1)*inform).transpose(0,1)
        else:
            raise Exception("Shape is not permmited.")

class PaddedBatchNorm1d(nn.Module):
    def __init__(self,channel_size):
        super().__init__()
        self.norm = nn.BatchNorm1d(channel_size)

    def forward(self,x,lengths):
        """N,C,L"""
        x = x.transpose(1,2)
        data = pack_padded_sequence(x,lengths, batch_first=True)
        padded_x,batch_sizes,_,_ = data
        normed_x = self.norm(padded_x)
        packed = PackedSequence(normed_x,batch_sizes)
        x = pad_packed_sequence(packed)[0]
        x = x.transpose(0,1).transpose(1,2)
        return x
    
class BasicModel(nn.Module,metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.out_channels = None
        self._distribution = {}

    @property
    def saved_distribution(self):
        return self._distribution

    def get_config(self):
        return {}

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()
            