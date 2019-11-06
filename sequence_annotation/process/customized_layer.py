from abc import ABCMeta
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F

class Concat(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self,x,lengths=None):
        #N,C,L
        min_length = min(item.shape[2] for item in x)
        x = [item[:,:,:min_length] for item in x]
        new_x = torch.cat(x,*self.args,**self.kwargs)
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
            self._epsilon = EPSILON
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

class PaddedNorm1d(nn.Module):
    def __init__(self,norm):
        super().__init__()
        self.norm = norm

    def forward(self,x,lengths):
        """N,C,L"""
        origin_length = x.shape[2]
        x = x.transpose(1,2)
        data = pack_padded_sequence(x,lengths, batch_first=True)
        padded_x,batch_sizes,_,_ = data
        normed_x = self.norm(padded_x)
        packed = PackedSequence(normed_x,batch_sizes)
        x = pad_packed_sequence(packed)[0]
        x = x.transpose(0,1).transpose(1,2)
        new_length = x.shape[2]
        pad_func = nn.ConstantPad1d((0,origin_length-new_length),0)
        x = pad_func(x)
        return x

class PaddedBatchNorm1d(nn.Module):
    def __init__(self,channel_size):
        super().__init__()
        self.norm = PaddedNorm1d(nn.BatchNorm1d(channel_size))

    def forward(self,x,lengths):
        """N,C,L"""
        x = self.norm(x,lengths)
        return x
    
class PaddedLayerNorm1d(nn.Module):
    def __init__(self,channel_size):
        super().__init__()
        self.norm = PaddedNorm1d(nn.LayerNorm(channel_size))

    def forward(self,x,lengths):
        """N,C,L"""
        x = self.norm(x,lengths)
        return x
    
class BasicModel(nn.Module,metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.in_channels = None
        self.out_channels = None
        self._distribution = {}

    @property
    def saved_distribution(self):
        return self._distribution

    def get_config(self):
        config = {}
        config['type'] = self.__class__.__name__
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        return config

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()
