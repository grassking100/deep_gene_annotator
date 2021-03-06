import os
from abc import ABCMeta
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence

EPSILON = 1e-32

class Concat:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        # N,C,L
        new_x = torch.cat(x, *self.args, **self.kwargs)
        return new_x

class PaddedNorm1d(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self._norm = norm

    def forward(self, x, lengths):
        """N,C,L"""
        origin_length = x.shape[2]
        x = x.transpose(1, 2)
        data = pack_padded_sequence(x, lengths, batch_first=True)
        padded_x, batch_sizes = data[:2]
        normed_x = self._norm(padded_x)
        packed = PackedSequence(normed_x, batch_sizes)
        x = pad_packed_sequence(packed)[0]
        x = x.permute(1, 2, 0)
        new_length = x.shape[2]
        if origin_length != new_length:
            pad_func = nn.ConstantPad1d((0, origin_length - new_length), 0)
            x = pad_func(x)
        return x


class BasicModel(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._in_channels = None
        self._out_channels = None
        self._save_distribution = False
        self._distribution = {}

    @property
    def in_channels(self):
        return self._in_channels
    
    @property
    def out_channels(self):
        return self._out_channels
        
    @property
    def save_distribution(self):
        return self._save_distribution

    @save_distribution.setter
    def save_distribution(self, value):
        self._save_distribution = value
        for module in self.children():
            if isinstance(module, BasicModel):
                module.save_distribution = value

    def _update_distribution(self, value, key=None):
        if self.save_distribution:
            if isinstance(value, dict):
                self._distribution.update(value)
            else:
                if key is None:
                    raise Exception(
                        "To save distribution of {}, you need to assign with its name"
                        .format(self.__class__.__name__))
                if isinstance(value, torch.Tensor):
                    value = value.cpu().detach().numpy()
                self._distribution[key] = value

    @property
    def saved_distribution(self):
        return self._distribution

    def get_config(self):
        config = {}
        config['type'] = self.__class__.__name__
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['save_distribution'] = self.save_distribution
        return config
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def reset(self):
        """Reset the state of all children if they have method reset"""
        for layer in self.children():
            if hasattr(layer, 'reset'):
                layer.reset()
                
    def load(self,path):
        weights = torch.load(path,map_location='cpu')
        self.load_state_dict(weights, strict=True)
        
    def save(self,path,overwrite=False):
        if os.path.exists(path) and not overwrite:
            raise Exception("Try to overwrite the existed weights to {}".format(path))
        torch.save(self.state_dict(), path)

class PaddedBatchNorm1d(BasicModel):
    def __init__(self, in_channels, affine=False, momentum=None,**kwrags):
        super().__init__()
        self._in_channels = self._out_channels = in_channels
        self._momentum = momentum or 0.1
        self._affine = affine
        self._norm = PaddedNorm1d(nn.BatchNorm1d(self.in_channels, affine=self._affine,
                                                 momentum=self._momentum,**kwrags))
        self._kwargs = kwrags

    def get_config(self):
        config = super().get_config()
        config['momentum'] = self._momentum
        config['affine'] = self._affine
        config['kwrags'] = self._kwargs
        return config

    def forward(self, x, lengths):
        """N,C,L"""
        N, C, L = x.shape
        if self.in_channels != C:
            raise Exception("Wrong channel size")
        x = self._norm(x, lengths)
        return x


NORM_CLASSES = {'PaddedBatchNorm1d': PaddedBatchNorm1d}

def generate_norm_class(norm_class=None, **kwargs):
    norm_class = norm_class or 'PaddedBatchNorm1d'
    def create(channel_size):
        if isinstance(norm_class, str):
            return NORM_CLASSES[norm_class](channel_size, **kwargs)
        else:
            return norm_class(channel_size, **kwargs)
    return create
