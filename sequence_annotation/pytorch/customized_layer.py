from abc import ABCMeta
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F

def init_GRU(gru):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

class Conv1d(nn.Conv1d):
    def forward(self,x,lengths=None):
        #N,C,L
        x = super().forward(x)
        if lengths is not None:
            new_lengths = [length - self.kernel_size[0] + 1 for length in lengths]
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

class PWM(nn.Module):
    def __init__(self,epsilon=None):
        super().__init__()
        self._epsilon = epsilon or 1e-32

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
            