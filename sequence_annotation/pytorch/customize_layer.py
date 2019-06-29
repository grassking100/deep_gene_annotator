from torch import nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn.init import uniform_,constant_
from .noisy_activation import NoisyReLU
import numpy as np
from .RNN.rnn import RNN

class Conv1d(nn.Conv1d):
    def forward(self,x,lengths=None):
        min_length = x.shape[2] - self.kernel_size[0]
        x = super().forward(x)
        if lengths is not None:
            new_lengths = []
            for length in lengths:
                new_lengths.append(min(length,min_length))
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
        min_length = None
        for item in x:
            length = item.shape[2]
            if min_length is None:
                min_length = length
            else:
                min_length = min(min_length,length)
        data_list = []
        for item in x:
            data_list.append(item.transpose(0,2)[:min_length].transpose(0,2))
        new_x = torch.cat(data_list,*self.args,**self.kwargs)
        if lengths is not None:
            new_lengths = []
            for length in lengths:
                new_lengths.append(min(length,min_length))
            return new_x, new_lengths
        else:
            return new_x

class GatedIndRnnCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self._gate_num = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_weights_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.gate_weights_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weights_i = nn.Parameter(torch.empty(hidden_size, input_size))
        self.i_gate_bias = nn.Parameter(torch.empty(hidden_size))
        self.f_gate_bias = nn.Parameter(torch.empty(hidden_size))
        self.input_bias = nn.Parameter(torch.empty(hidden_size))
        self.gate_function =  NoisyHardSigmoid()
        self.ln = nn.LayerNorm(hidden_size)
        self.recurrent_function = NoisyReLU()
        self.reset_parameters()
        self.output_names = ['new_h','pre_gate_f','pre_gate_i','gate_f','gate_i','values_i','pre_h']
    def reset_parameters(self):
        gate_bound_ih = (1/((self.input_size**0.5)))
        gate_bound_hh = (1/((self.input_size**0.5)))
        input_bound = (1/(self.input_size**0.5))
        uniform_(self.gate_weights_ih,-gate_bound_ih,gate_bound_ih)
        uniform_(self.gate_weights_hh,-gate_bound_hh,gate_bound_hh)
        uniform_(self.weights_i,-input_bound,input_bound)
        constant_(self.i_gate_bias,1)
        constant_(self.f_gate_bias,1)
        constant_(self.input_bias,0)
    def forward(self, input, state):
        #input shape should be (number,feature size)
        values_i = F.linear(input, self.weights_i,self.input_bias)
        pre_gate_ih = F.linear(input, self.gate_weights_ih)
        pre_gate_hh = F.linear(self.ln(state), self.gate_weights_hh)
        pre_gate_f = pre_gate_ih + self.f_gate_bias
        pre_gate_i = pre_gate_hh + self.i_gate_bias
        gate_f = self.gate_function(pre_gate_f,training=self.training)
        gate_i = self.gate_function(pre_gate_i,training=self.training)
        pre_h = state*gate_f+ values_i*gate_i
        new_h = self.recurrent_function(pre_h,self.training)
        return new_h,pre_gate_f,pre_gate_i,gate_f,gate_i,values_i,pre_h

class PWM(nn.Module):
    def __init__(self,epsilon=None):
        super().__init__()
        self._epsilon = epsilon or 1e-32
    def reset_parameters(self):
        pass
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

class ConcatCNN(nn.Module):
    def __init__(self,input_size,num_layers,kernel_sizes,out_channels,
                 ln_mode=None,with_pwm=None):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.with_pwm = with_pwm or False
        
        if ln_mode in ["before_cnn","after_cnn","after_activation",None]:
            self.ln_mode = ln_mode
        else:
            raise Exception('ln_mode should be "before_cnn", "after_cnn", None, or "after_activation"')
        self.activation_function = NoisyReLU()
        self._build_layers()
        self.reset_parameters()
    def _build_layers(self):
        in_num = self.input_size
        self.lns = []
        self.cnns = []
        self.pwm = PWM()
        use_bias = self.ln_mode != "after_cnn"
        for index in range(self.num_layers):
            kernel_size = self.kernel_sizes[index]
            out_channels = self.out_channels[index]
            cnn = Conv1d(in_channels=in_num,kernel_size=kernel_size,
                         out_channels=out_channels,bias=use_bias)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            if self.ln_mode is not None:
                in_channel = {"before_cnn":in_num,"after_cnn":out_channels,"after_activation":out_channels}
                ln = nn.LayerNorm(in_channel[self.ln_mode])
                self.lns.append(ln)
                setattr(self, 'ln_'+str(index), ln)
            in_num += out_channels
        self.hidden_size = in_num
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()
    def _layer_normalized(self,ln,x,name,distribution_output):
        x = ln(x.transpose(1, 2)).transpose(1, 2)
        distribution_output[name] = x
        return x
    def forward(self, x, lengths):
        #X shape : N,C,L
        distribution_output = {}
        x_ = x
        for index in range(self.num_layers):
            pre_x = x_
            cnn = self.cnns[index]
            if self.ln_mode=='before_cnn':
                ln = self.lns[index]
                x_ = self._layer_normalized(ln,x_,'ln_'+str(index),distribution_output)
                if self.with_pwm:
                    x_ = self.pwm(x_)
            x_,lengths = cnn(x_,lengths)
            distribution_output['cnn_x_'+str(index)] = x_
            if self.ln_mode=='after_cnn':
                ln = self.lns[index]
                x_ = self._layer_normalized(ln,x_,'ln_'+str(index),distribution_output)
                if self.with_pwm:
                    x_ = self.pwm(x_)
            x_ = self.activation_function(x_)
            distribution_output['post_act_x_'+str(index)] = x_
            if self.ln_mode=='after_activation':
                ln = self.lns[index]
                x_ = self._layer_normalized(ln,x_,'ln_'+str(index),distribution_output)
                if self.with_pwm:
                    x_ = self.pwm(x_)
            x_,lengths = Concat(dim=1)([pre_x,x_],lengths)
        distribution_output['cnn_result'] = x_
        return x_,lengths,distribution_output
    
class ConcatRNN(nn.Module):
    def __init__(self,input_size,num_layers,hidden_size,rnn_cell_class,
                 rnns_as_output=True,layer_norm=True,bidirectional=True,
                 tanh_after_rnn=False,cnn_before_rnn=False,**kwargs):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layer_norm = layer_norm
        self.kwargs = kwargs
        self.rnn_cell_class = rnn_cell_class
        self.rnns_as_output = rnns_as_output
        self.tanh_after_rnn = tanh_after_rnn
        self.cnn_before_rnn = cnn_before_rnn
        self.bidirectional = bidirectional
        self._build_layers()
        self.reset_parameters()
    def _build_layers(self):
        in_num = self.input_size
        self.rnns = []
        self.cnns_before_rnn = []
        self.lns=[]
        for index in range(self.num_layers):
            hidden_size = self.hidden_size[index]
            rnn_cell = self.rnn_cell_class(input_size=in_num,hidden_size=hidden_size)
            rnn = RNN(rnn_cell,bidirectional=self.bidirectional,**self.kwargs)
            self.rnns.append(rnn)
            setattr(self, 'rnn_'+str(index), rnn)
            if self.cnn_before_rnn:
                cbr = Conv1d(in_channels=in_num,kernel_size=2,out_channels=in_num)
                self.cnns_before_rnn.append(cbr)
                setattr(self, 'cbr_'+str(index), cbr)
            if self.layer_norm:
                ln = nn.LayerNorm(hidden_size*2)
                self.lns.append(ln)
                setattr(self, 'ln_'+str(index), ln)
            in_num += hidden_size*2
        if self.rnns_as_output:
            self.hidden_size = hidden_size*2*self.num_layers
        else:
            self.hidden_size = in_num
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()
    def forward(self, x, lengths):
        #X shape : N,C,L
        x_ = x.transpose(1, 2)
        distribution_output = {}
        rnn_output = {}
        for name in self.rnns[0].output_names:
            if name not in rnn_output.keys():
                rnn_output[name] = []
        rnn_output['after_rnn'] = []
        for index in range(self.num_layers):
            rnn = self.rnns[index]
            pre_x = x_
            if self.cnn_before_rnn:
                cbr = self.cnns_before_rnn[index]
                x_=x_.transpose(1, 2)
                distribution_output["pre_cbr_"+str(index)] = x_
                x_,lengths = cbr(x_,lengths)
                distribution_output["cbr_"+str(index)] = x_
                x_=x_.transpose(1, 2)
            x_ = rnn(x_,lengths)
            if hasattr(rnn,'output_names'):
                for name,item in zip(rnn.output_names,x_):
                    rnn_output[name].append(item)
                    values = item.transpose(1,2)
                    distribution_output["rnn_"+str(index)+"_"+name] = values
                x_ = x_[rnn.output_names.index('new_h')]
            if self.layer_norm:
                x_ = self.lns[index](x_)
            if self.tanh_after_rnn:
                x_ = x_.tanh()
            rnn_output['after_rnn'].append(x_)
            x_,lengths = Concat(dim=1)([pre_x.transpose(1,2),x_.transpose(1,2)],lengths)
            x_ = x_.transpose(1,2)
        for name,value in rnn_output.items():
            value = [item.transpose(1,2) for item in value]
            temp,lengths = Concat(dim=1)(value,lengths)
            rnn_output[name] = temp
        if self.rnns_as_output:
            x = rnn_output['after_rnn']
        else:
            x = x_.transpose(1, 2)
        distribution_output['rnn_result'] = rnn_output['after_rnn']
        return x,lengths,distribution_output
