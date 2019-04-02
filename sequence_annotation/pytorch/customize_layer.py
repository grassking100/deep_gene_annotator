import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import ones_,zeros_,uniform_,normal_,constant_,eye_
from torch.nn import Hardtanh, Sigmoid,Tanh,ReLU
from torch import randn
import numpy as np
import time
from abc import abstractproperty,ABCMeta
from .RNN.rnn import RNN
from .CRF import BatchCRF

hard_sigmoid = Hardtanh(min_val=0)
def sgn(x):
    return (x>0).float()*2-1

class NoisyHardAct(nn.Module):
    def __init__(self,alpha=None,c=None,p=None):
        super().__init__()
        self._alpha = alpha or 1
        self._c = c or 0.05
        self._p = p or 1
        self._sigmoid = Sigmoid()
        self._hard_function = self._get_hard_function
        self._alpha_complement = 1-self._alpha
        if self._alpha_complement>0:
            self._sgn=1
        else:
            self._sgn=-1
    @abstractproperty
    def _get_hard_function(self):
        pass
    def forward(self,x):
        h = self._hard_function(x)
        if self.training:
            random = torch.abs(torch.randn_like(x))
            diff = h-x
            d = -sgn(x)*self._sgn
            if self._alpha == 1:
                native_result = h
            else:
                native_result = self._alpha*h+self._alpha_complement*x
            if self._p == 1:
                diff = self._p*diff
            sigma = self._c*(self._sigmoid(diff)-0.5)**2
            return native_result+(d*sigma*random)
        else:
            return h

class NoisyHardTanh(NoisyHardAct):
    @property
    def _get_hard_function(self):
        return Hardtanh()

class NoisyHardSigmoid(NoisyHardAct):
    @property
    def _get_hard_function(self):
        return hard_sigmoid

class NoisyReLU(NoisyHardAct):
    @property
    def _get_hard_function(self):
        return ReLU()

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

def init_GRU(gru):
    for name, param in gru.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

class IndLinear(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.vector = torch.nn.Parameter(torch.empty(size),requires_grad=True)
        self.bias = torch.nn.Parameter(torch.empty(size),requires_grad=True)
        self.reset_parameters()
    def forward(self,x):
        return self.vector*x+self.bias
    def reset_parameters(self):
        uniform_(self.vector,-1,1)
        constant_(self.bias,0)

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

class SeqAnnLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 0
        self.ignore_index = -1
        self.alphas = None

    def forward(self, output, target,spatial_weights=None,**kwargs):
        """data shape is N,C,L"""
        if isinstance(output,tuple):
            pt = output[0]
            pt,_,distribution_output,_ = output
        else:
            pt = output
        input_length = pt.shape[2]
        if isinstance(output,tuple):
            gate_i = distribution_output['new_h']
            center=0.5
            gate_loss_ = ((center**2)-(gate_i-center)**2).mean(1)
        if list(pt.shape) != list(target.shape):
            target = target.transpose(0,2)[:input_length].transpose(0,2)
        target = target.float()
        if self.ignore_index is not None:
            mask = (target.max(1)[0] != self.ignore_index).float()
        decay_cooef = (1-pt)**self.gamma
        loss_ =  -decay_cooef* (pt+1e-32).log() * target
        if spatial_weights is not None:
            if list(pt.shape) != list(spatial_weights.shape):
                 spatial_weights = spatial_weights.transpose(0,2)[:input_length].transpose(0,2)
            spatial_weights = torch.FloatTensor(spatial_weights)
        if self.alphas is not None:
            loss_ = (loss_.transpose(1,2)*self.alphas).transpose(1,2)
        loss_ = loss_.sum(1)
        if self.ignore_index is not None:
            loss_ = loss_*mask
            loss = loss_.sum()/mask.sum()
            if isinstance(output,tuple):
                gate_loss_ = gate_loss_*mask
                gate_loss = gate_loss_.sum()/mask.sum()
                loss+=gate_loss
        else:
            loss = loss_.mean()
            if isinstance(output,tuple):
                gate_loss = gate_loss_.mean()
                loss+=gate_loss
        return loss

class ConcatCNN(nn.Module):
    def __init__(self,in_channels,layer_num,kernel_sizes,out_channels,
                 ln_mode=None,with_pwm=True):
        super().__init__()
        self.in_channels = in_channels
        self.layer_num = layer_num
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.with_pwm = with_pwm
        if ln_mode in ["before_cnn","after_cnn","after_activation",None]:
            self.ln_mode = ln_mode
        else:
            raise Exception('ln_mode should be "before_cnn", "after_cnn", None, or "after_activation"')
        self.activation_function = NoisyReLU()
        self._build_layers()
        self.reset_parameters()
    def _build_layers(self):
        in_num = self.in_channels
        self.lns = []
        self.cnns = []
        self.pwm = PWM()
        for index in range(self.layer_num):
            kernel_size = self.kernel_sizes[index]
            out_channels = self.out_channels[index]
            cnn = Conv1d(in_channels=in_num,kernel_size=kernel_size,
                            out_channels=out_channels)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            if self.ln_mode is not None:
                in_channel = {"before_cnn":in_num,"after_cnn":out_channels,"after_activation":out_channels}
                ln = nn.LayerNorm(in_channel[self.ln_mode])
                self.lns.append(ln)
                setattr(self, 'ln_'+str(index), ln)
            in_num += out_channels
        self.out_channels = in_num
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
        for index in range(self.layer_num):
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
    def __init__(self,in_channels,layer_num,hidden_sizes,rnn_cell_class,
                 rnns_as_output=True,layer_norm=True,bidirectional=True,
                 tanh_after_rnn=False,cnn_before_rnn=False,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.layer_num = layer_num
        self.hidden_sizes = hidden_sizes
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
        in_num = self.in_channels
        self.rnns = []
        self.cnns_before_rnn = []
        self.lns=[]
        for index in range(self.layer_num):
            hidden_size = self.hidden_sizes[index]
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
            self.out_channels = hidden_size*2*self.layer_num
        else:
            self.out_channels = in_num
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
        for index in range(self.layer_num):
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

class ISeqAnnModel(nn.Module,metaclass=ABCMeta):
    @abstractproperty
    def saved_lengths(self):
        pass
    @abstractproperty
    def saved_distribution(self):
        pass
    @abstractproperty
    def saved_outputs(self):
        pass
    def saved_onehot_outputs(self):
        pass

class SeqAnnModel(ISeqAnnModel):
    def __init__(self,in_channels,out_channels,
                 cnns_setting=None,
                 rnns_setting=None,
                 reduce_cnn_ratio=None,
                 pwm_before_rnns=None,
                 use_CRF=None,
                 reduced_cnn_number=None,
                 last_kernel_size=None):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels = out_channels
        self.cnns_setting = cnns_setting or {'layer_num':0}
        self.rnns_setting = rnns_setting or {'layer_num':0}
        self.cnn_num = self.cnns_setting['layer_num']
        self.rnn_num = self.rnns_setting['layer_num']
        self.reduced_cnn_number = reduced_cnn_number
        self.pwm_before_rnns = pwm_before_rnns or True
        self.last_kernel_size = last_kernel_size or 1
        self.use_CRF = use_CRF or False
        self.reduce_cnn_ratio = reduce_cnn_ratio or 1
        if self.reduce_cnn_ratio <= 0:
            raise Exception("Reduce_cnn_ratio should be larger than zero")
        self.CRF = None
        self._saved_index_outputs = None
        self._outputs = None
        self._lengths = None
        self._distribution = None
        self._reduce_cnn = self.reduce_cnn_ratio < 1 or self.reduced_cnn_number is not None
        self._build_layers()
        self.reset_parameters()
        self.write_cnn = True
        self.write_rnn = True
    def _build_layers(self):
        in_channels=self.in_channels
        if self.cnn_num > 0:
            self.cnns = ConcatCNN(in_channels,**self.cnns_setting)
            self.cnn_ln = nn.LayerNorm(self.cnns.out_channels)
            if self._reduce_cnn:
                if self.reduce_cnn_ratio < 1:
                    out_channels = int(self.cnns.out_channels*self.reduce_cnn_ratio)
                else:
                    out_channels = self.reduced_cnn_number
                self.cnn_merge = Conv1d(in_channels=self.cnns.out_channels,
                                        kernel_size=1,out_channels=out_channels)
                in_channels = out_channels    
            else:
                in_channels = self.cnns.out_channels
            if self.pwm_before_rnns:
                self.cnn_pwm = PWM()
        if self.rnn_num > 0:
            print(in_channels)
            self.rnns = ConcatRNN(in_channels,**self.rnns_setting)
            in_channels=self.rnns.out_channels
        self.last = Conv1d(in_channels=in_channels,kernel_size=self.last_kernel_size,
                              out_channels=self.out_channels)
        self.CRF = BatchCRF(self.out_channels)
       
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()
        #if self._reduce_cnn:
        #    normal_(self.cnn_merge.weight)
        #    constant_(self.cnn_merge.bias,0)
        constant_(self.last.bias,0)
        normal_(self.last.weight)

    def forward(self, x, lengths,return_values=False):
        #X shape : N,C,L
        figrue_output = {}
        distribution_output = {}
        if self.cnn_num > 0:
            pre_time = time.time()
            x,lengths,cnn_distribution = self.cnns(x,lengths)
            x = self.cnn_ln(x.transpose(1,2)).transpose(1,2)
            if self._reduce_cnn:
                x,lengths = self.cnn_merge(x,lengths)
            if self.pwm_before_rnns:
                x = self.cnn_pwm(x)
            if self.write_cnn:
                distribution_output.update(cnn_distribution)
            print("CNN time",int(time.time()-pre_time))
        if self.rnn_num > 0:
            pre_time = time.time()
            distribution_output["pre_rnn_result"] = x
            x,lengths,rnn_distribution =self.rnns(x,lengths)
            distribution_output["pre_last_result"] = x
            if self.write_rnn:
                distribution_output.update(rnn_distribution)
            print("RNN time",int(time.time()-pre_time))
        x,lengths = self.last(x,lengths)
        distribution_output["last"] = x
        if not self.use_CRF:
            x = nn.LogSoftmax(dim=1)(x)
            distribution_output["log_softmax"] = x
            x = x.exp()
            distribution_output["softmax"] = x
        self._distribution = distribution_output
        self._lengths = lengths
        self._outputs = x
        if not self.use_CRF:
            output_index = self._outputs.max(1)[1]
        else:
            output_index = self.CRF(self._outputs,self._lengths)
        self._saved_index_outputs = output_index 
        return x
    @property
    def saved_outputs(self):
        return self._outputs
    @property
    def saved_lengths(self):
        return self._lengths
    @property
    def saved_distribution(self):
        return self._distribution
    @property
    def saved_index_outputs(self):
        if self._saved_index_outputs is not None:
            return self._saved_index_outputs
        else:
            raise Exception("Object must run forward at least one times")