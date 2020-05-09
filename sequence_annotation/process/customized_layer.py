from abc import ABCMeta
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F

EPSILON = 1e-32

class Concat:
    def __init__(self, handle_length=True, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.handle_length = handle_length

    def __call__(self, x):
        # N,C,L
        if self.handle_length:
            min_length = min(item.shape[2] for item in x)
            x = [item[:, :, :min_length] for item in x]
        new_x = torch.cat(x, *self.args, **self.kwargs)
        return new_x


class Add:
    def __init__(self, handle_length=True):
        self.handle_length = handle_length

    def __call__(self, lhs, rhs):
        # N,C,L
        if self.handle_length:
            min_length = min(item.shape[2] for item in [lhs, rhs])
            lhs = lhs[:, :, :min_length]
            rhs = rhs[:, :, :min_length]
        new_x = lhs + rhs
        return new_x


class PWM(nn.Module):
    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is None:
            self._epsilon = EPSILON
        else:
            self._epsilon = epsilon

    def forward(self, x):
        # N,C,L
        channel_size = x.shape[1]
        freq = F.softmax(x, dim=1)
        freq_with_background = freq * channel_size
        inform = (freq * (freq_with_background + self._epsilon).log2_()).sum(1)
        if len(x.shape) == 3:
            return (freq.transpose(0, 1) * inform).transpose(0, 1)
        elif len(x.shape) == 2:
            return (freq.transpose(0, 1) * inform).transpose(0, 1)
        else:
            raise Exception("Shape is not permmited.")


class PaddedNorm1d(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x, lengths):
        """N,C,L"""
        origin_length = x.shape[2]
        x = x.transpose(1, 2)
        data = pack_padded_sequence(x, lengths, batch_first=True)
        padded_x, batch_sizes, _, _ = data
        normed_x = self.norm(padded_x)
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
        self.in_channels = None
        self.out_channels = None
        self._save_distribution = True
        self._distribution = {}

    @property
    def save_distribution(self):
        return self._save_distribution

    @save_distribution.setter
    def save_distribution(self, value):
        self._save_distribution = value
        for module in self.children():
            if isinstance(module, BasicModel):
                module.save_distribution = value

    def update_distribution(self, value, key=None):
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

class PaddedBatchNorm1d(BasicModel):
    def __init__(self, in_channels, **kwrags):
        super().__init__()
        self.in_channels = self.out_channels = in_channels
        self.norm = PaddedNorm1d(nn.BatchNorm1d(self.in_channels, **kwrags))
        self._kwargs = kwrags

    def get_config(self):
        config = super().get_config()
        config['kwrags'] = self._kwargs
        return config

    def forward(self, x, lengths):
        """N,C,L"""
        N, C, L = x.shape
        if self.in_channels != C:
            raise Exception("Wrong channel size")
        x = self.norm(x, lengths)
        return x


NORM_CLASS = {'PaddedBatchNorm1d': PaddedBatchNorm1d}


def generate_norm_class(norm_class, **kwargs):
    def create(channel_size):
        if isinstance(norm_class, str):
            return NORM_CLASS[norm_class](channel_size, **kwargs)
        else:
            return norm_class(channel_size, **kwargs)

    return create
