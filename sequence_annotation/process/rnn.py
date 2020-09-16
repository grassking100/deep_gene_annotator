from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .customized_layer import BasicModel
from .cnn import Conv1d
from .utils import xavier_uniform_extend_
from torch.nn.init import zeros_


def separate_xav_rnn_init(rnn, mode=None, chunk_size=None):
    chunk_size = chunk_size or 3
    mode = mode or 'fan_in'
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            weights = []
            weight_ih_name = 'weight_ih_l{}{}'.format(index, suffix)
            weight_hh_name = 'weight_hh_l{}{}'.format(index, suffix)
            weight_ih = getattr(rnn,weight_ih_name).chunk(chunk_size, 0)
            weight_hh = getattr(rnn,weight_hh_name).chunk(chunk_size, 0)
            weights += weight_ih
            weights += weight_hh
            for var in weights:
                xavier_uniform_extend_(var, mode=mode)


def xav_rnn_init(rnn, mode=None, chunk_size=None):
    chunk_size = chunk_size or 3
    mode = mode or 'fan_in'
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            weights = []
            weight_ih_name = 'weight_ih_l{}{}'.format(index, suffix)
            weight_hh_name = 'weight_hh_l{}{}'.format(index, suffix)
            weight_ih = getattr(rnn,weight_ih_name).chunk(chunk_size, 0)
            weight_hh = getattr(rnn,weight_hh_name).chunk(chunk_size, 0)
            weights += weight_ih
            weights += weight_hh
            i_o, i_i = weight_ih[0].shape
            h_o, h_i = weight_hh[0].shape
            if i_o != h_o:
                raise Exception()
            if mode == 'fan_in':
                n = i_i + h_i
            elif mode == 'fan_out':
                n = i_o
            elif mode == 'both':
                n = (i_o + i_i + h_i) / 2
            else:
                raise Exception()
            for var in weights:
                xavier_uniform_extend_(var, n=n)


def bias_zero_rnn_init(rnn, chunk_size=None):
    chunk_size = chunk_size or 3
    suffice = ['']
    if rnn.bidirectional:
        suffice.append('_reverse')
    for index in range(rnn.num_layers):
        for suffix in suffice:
            weights = []
            weight_ih_name = 'weight_ih_l{}{}'.format(index, suffix)
            weight_hh_name = 'weight_hh_l{}{}'.format(index, suffix)
            weights += getattr(rnn, weight_ih_name).chunk(chunk_size, 0)
            weights += getattr(rnn, weight_hh_name).chunk(chunk_size, 0)
            for var in weights:
                zeros_(var)


def xav_bias_zero_rnn_init(rnn, mode=None, chunk_size=None):
    xav_rnn_init(rnn, mode, chunk_size)
    bias_zero_rnn_init(rnn, chunk_size)


def separate_xav_bias_zero_rnn_init(rnn, mode=None, chunk_size=None):
    separate_xav_rnn_init(rnn, mode, chunk_size)
    bias_zero_rnn_init(rnn, chunk_size)


def in_xav_bias_zero_gru_init(rnn):
    return xav_bias_zero_rnn_init(rnn, 'fan_in')


def out_xav_bias_zero_gru_init(rnn):
    return xav_bias_zero_rnn_init(rnn, 'fan_out')


def separate_in_xav_bias_zero_gru_init(rnn):
    return separate_xav_bias_zero_rnn_init(rnn, 'fan_in')


def separate_out_xav_bias_zero_gru_init(rnn):
    return separate_xav_bias_zero_rnn_init(rnn, 'fan_out')


RNN_INIT_MODE = {
    None: None,
    'in_xav_bias_zero_gru_init': in_xav_bias_zero_gru_init,
    'out_xav_bias_zero_gru_init': out_xav_bias_zero_gru_init,
    'separate_in_xav_bias_zero_gru_init': separate_in_xav_bias_zero_gru_init,
    'separate_out_xav_bias_zero_gru_init': separate_out_xav_bias_zero_gru_init,
}


def _forward(rnn, x, lengths, state=None, batch_first=True):
    # Input:N,C,L, Output: N,C,L
    N,C,L = x.shape
    hidden_size = rnn.hidden_size
    num_layers = rnn.num_layers
    if rnn.bidirectional:
        n_direction = 2
    else:
        n_direction = 1
    if not batch_first:
        x = x.transpose(0, 1)
    x = x.transpose(1, 2)
    x = pack_padded_sequence(x, lengths, batch_first=batch_first)
    x,hidden_states = rnn(x, state)
    hidden_states = hidden_states.view(num_layers,n_direction,N,hidden_size)    
    x, _ = pad_packed_sequence(x, batch_first=batch_first)
    x = x.transpose(1, 2)
    if not batch_first:
        x = x.transpose(0, 1)
    return x,hidden_states


class _RNN(BasicModel):
    def __init__(self, in_channels, hidden_size, customized_init=None,
                 bidirectional=True,batch_first=True,**kwargs):
        super().__init__()
        if isinstance(customized_init, str):
            customized_init = RNN_INIT_MODE[customized_init]
        self._customized_init = customized_init or RNN_INIT_MODE['in_xav_bias_zero_gru_init']
        self._rnn = self._create_rnn(in_channels, hidden_size,
                                     bidirectional=bidirectional,
                                     batch_first=batch_first,**kwargs)
        self._kwargs = kwargs
        self._in_channels = in_channels
        self._out_channels = self._rnn.hidden_size
        self._hidden_size = hidden_size
        if self._rnn.bidirectional:
            self._out_channels *= 2
        self._dropout = self._rnn.dropout
        self.reset_parameters()
        
    @property
    def dropout(self):
        return self._dropout
        
    @abstractmethod
    def _create_rnn(self, in_channels, hidden_size, **kwargs):
        pass

    def get_config(self):
        config = super().get_config()
        config['kwargs'] = self._kwargs
        config['hidden_size'] = self._hidden_size
        config['customized_init'] = str(self._customized_init)
        config['bidirectional'] = self._rnn.bidirectional
        config['batch_first'] = self._rnn.batch_first
        return config

    def reset_parameters(self):
        super().reset_parameters()
        if self._customized_init is not None:
            self._customized_init(self._rnn)


class GRU(_RNN):
    def _create_rnn(self, in_channels, hidden_size,**kwargs):
        return nn.GRU(in_channels, hidden_size,**kwargs)

    def forward(self, x, lengths, **kwargs):
        x,hidden_states = _forward(self._rnn, x, lengths, batch_first=self._rnn.batch_first)        
        return x


class LSTM(_RNN):
    def _create_rnn(self, in_channels, hidden_size,**kwargs):
        return nn.LSTM(in_channels, hidden_size,**kwargs)

    def forward(self, x, lengths, **kwargs):
        x = _forward(self._rnn, x, lengths, batch_first=self._rnn.batch_first)
        return x


class ProjectedRNN(BasicModel):
    def __init__(self,in_channels,hidden_size, out_channels,norm_class=None,
                 customized_cnn_init=None,customized_rnn_init=None,
                 name=None,output_act=None,rnn_class=None,
                 dropout_before_rnn=False,**kwargs):
        super().__init__()
        self._output_act = None
        if output_act is not None:
            if not isinstance(output_act,str):
                self._output_act = output_act
            elif output_act == 'softmax':
                self._output_act = torch.nn.Softmax(dim=1)
            elif output_act == 'sigmoid':
                self._output_act = torch.nn.Sigmoid()
            else:
                raise Exception("Wrong activation name, {}".format(output_act))
        
        self._dropout_before_rnn = dropout_before_rnn
        if name is None or name == '':
            self._name = 'rnn'
        else:
            self._name = "{}_rnn".format(name)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._rnn_class = rnn_class or GRU
        self._rnn = self._rnn_class(self.in_channels,hidden_size,
                                    customized_init=customized_rnn_init,**kwargs)
        self._project = Conv1d(self._rnn.out_channels,self.out_channels,1,
                              customized_init=customized_cnn_init)
        self._norm = None
        if norm_class is not None:
            self._norm = norm_class(in_channels)
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config['rnn_class'] = self._rnn_class.__name__
        config['rnn'] = self._rnn.get_config()
        config['project'] = self._project.get_config()
        config['name'] = self._name
        config['output_act'] = str(self._output_act)
        config['dropout_before_rnn'] = self._dropout_before_rnn
        if self._norm is not None:
            config['norm'] = self._norm.get_config()
        return config

    def forward(self, x, lengths, return_intermediate=False, **kwargs):
        if self._rnn.dropout > 0 and self.training and self._dropout_before_rnn:
            x = F.dropout(x, self._dropout, self.training)
            
        if self._norm is not None:
            x = self._norm(x,lengths)
        post_rnn = self._rnn(x, lengths)
        self._update_distribution(post_rnn, key=self._name)

        if self._rnn.dropout > 0 and self.training:
            post_rnn = F.dropout(post_rnn, self._dropout, self.training)
        result, lengths = self._project(post_rnn, lengths=lengths)[:2]

        if self._output_act is not None:
            result = self._output_act(result)

        if return_intermediate:
            return result, post_rnn
        else:
            return result


RNN_CLASSES = {'GRU': GRU, 'LSTM': LSTM, 'ProjectedRNN': ProjectedRNN}
