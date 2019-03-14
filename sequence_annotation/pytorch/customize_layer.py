import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.init import ones_,zeros_,uniform_,normal_,constant_,eye_
from torch.nn import Hardtanh, Sigmoid
from torch import randn
import numpy as np
from .RNN.rnn import RNN
import time
from .CRF import BatchCRF
from abc import abstractproperty,ABCMeta
def sgn(x):
    return (x>0).float()*2-1

def noisy_hard_function(hard_function,alpha=1.0,c=0.05,p=1):
    sigmoid = Sigmoid()
    def noised(x,training):
        h = hard_function(x)
        if training:
            native_result = (alpha*h+(1-alpha)*x).float()
            diff = h-x
            temp = sgn(x)
            temp2=sgn(torch.FloatTensor([1-alpha])).cuda()
            d = (-temp*temp2)
            sigma=(c*(sigmoid(p*diff)-0.5)**2)
            random = torch.abs(randn(x.size())).cuda()
            return native_result+(d*sigma*random)
        else:
            return h
    return noised

def noisy_hard_tanh(alpha=1.0,c=0.05,p=1):
    hard_tanh = Hardtanh()
    return noisy_hard_function(hard_tanh,alpha,c,p)

def noisy_hard_sigmoid(alpha=1.0,c=0.05,p=1):
    hard_sigmoid = Hardtanh(0,1)
    return noisy_hard_function(hard_sigmoid,alpha,c,p)

def noisy_relu(alpha=1.0,c=0.05,p=1):
    return noisy_hard_function(F.relu,alpha,c,p)

class PadConv1d(nn.Module):
    def __init__(self,pad_value=None,*args,**kwargs):
        super().__init__()
        self._kernel_size = kwargs['kernel_size']
        self._in_channels = kwargs['in_channels']
        self.cnn = nn.Conv1d(*args,**kwargs).cuda()
        self.weight=cnn.weight
        self.bias=cnn.bias
        self.pad_value = pad_value
        self.reset_parameters()
    def reset_parameters(self):
        bound = (1/(self._in_channels*self._kernel_size))
        uniform_(self.cnn.weight,-bound,bound)
        if self.cnn.bias is not None:
            constant_(self.cnn.bias,0)
    def forward(self,x):
        if self.pad_value is not None:
            x = F.pad(x, [0,self._kernel_size-1], 'constant', self.pad_value)
        x = self.cnn(x)
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

class GatedIndRnnCellOld(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self._gate_num = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_weights = torch.nn.Parameter(torch.empty(hidden_size, hidden_size+input_size))
        self.weights_i = torch.nn.Parameter(torch.empty(hidden_size, input_size))
        self.gate_bias = torch.nn.Parameter(torch.empty(hidden_size))
        self.bn = nn.LayerNorm(hidden_size+ input_size).cuda()
        self.gate_function =  noisy_hard_sigmoid()
        self.recurrent_function = noisy_relu(c=1)
        self.reset_parameters()
        self.output_names = ['new_h','concat_input','pre_gate_i','gate_i','values_i','pre_h']
    def reset_parameters(self):
        gate_bound = (1/(self.input_size+self.hidden_size))
        input_bound = (1/(self.input_size))
        uniform_(self.gate_weights,-gate_bound,gate_bound)
        uniform_(self.weights_i,-input_bound,input_bound)
        constant_(self.gate_bias,0.5)
    def forward(self, input, state):
        #input shape should be (number,feature size)
        concat_input = torch.cat([input,state], dim=1)
        pre_gate_i = F.linear(concat_input, self.gate_weights,self.gate_bias)
        gate_i = self.gate_function(pre_gate_i,training=self.training)
        values_i = F.linear(input, self.weights_i)
        values_i = Tanh()(values_i)
        pre_h = state*(1-gate_i)+ values_i*gate_i
        new_h = self.recurrent_function(pre_h,training=self.training)
        return new_h,concat_input,pre_gate_i,gate_i,values_i,pre_h

class GatedIndRnnCellOld2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self._gate_num = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_weights_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.gate_weights_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weights_i = nn.Parameter(torch.empty(hidden_size, input_size))
        self.gate_bias = nn.Parameter(torch.empty(hidden_size))
        self.input_bias = nn.Parameter(torch.empty(hidden_size))
        self.gate_function =  noisy_hard_sigmoid()
        self.recurrent_function = noisy_hard_sigmoid()
        self.reset_parameters()
        self.output_names = ['new_h','pre_gate_i','gate_i','values_i','pre_h',
                             'pre_values_i','pre_gate_i_ih','pre_gate_i_hh']
    def reset_parameters(self):
        gate_bound_hh = (1/((self.hidden_size)))
        gate_bound_ih = (1/((self.input_size)))
        input_bound = (1/(self.input_size))
        #eye_(self.gate_weights_hh)
        uniform_(self.gate_weights_hh,-gate_bound_hh,gate_bound_hh)
        uniform_(self.gate_weights_ih,-gate_bound_ih,gate_bound_ih)
        uniform_(self.weights_i,-input_bound,input_bound)
        constant_(self.gate_bias,0.5)
        constant_(self.input_bias,0)
    def forward(self, input, state):
        #input shape should be (number,feature size)
        concat_input = torch.cat([input,state], dim=1)
        pre_gate_i_ih = F.linear(input, self.gate_weights_ih)
        pre_gate_i_hh = F.linear(state, self.gate_weights_hh)
        pre_gate_i = pre_gate_i_ih + pre_gate_i_hh +self.gate_bias
        gate_i = self.gate_function(pre_gate_i,training=self.training)
        pre_values_i = F.linear(input, self.weights_i,self.input_bias)
        values_i = noisy_hard_sigmoid(c=1)(pre_values_i,training=self.training)
        pre_h = state*gate_i+ values_i*(1-gate_i)
        #!!!To avoid NAN problem
        #if self.training:
        #    new_h = self.recurrent_function(pre_h,training=self.training)
        #else:
        new_h = pre_h
        return new_h,pre_gate_i,gate_i,values_i,pre_h,pre_values_i,pre_gate_i_ih,pre_gate_i_hh

class GatedIndRnnCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self._gate_num = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_weights_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weights_i = nn.Parameter(torch.empty(hidden_size, input_size))
        self.gate_bias = nn.Parameter(torch.empty(hidden_size))
        self.input_bias = nn.Parameter(torch.empty(hidden_size))
        self.gate_function =  noisy_hard_sigmoid(alpha=1)
        self.recurrent_function = noisy_relu(alpha=1)
        self.reset_parameters()
        self.output_names = ['new_h','pre_gate','gate','values_i','pre_h','pre_gate_i_ih']
    def reset_parameters(self):
        gate_bound_ih = (1/((self.input_size)))
        input_bound = (1/(self.input_size))
        uniform_(self.gate_weights_ih,-gate_bound_ih,gate_bound_ih)
        uniform_(self.weights_i,-input_bound,input_bound)
        constant_(self.gate_bias,1)
        constant_(self.input_bias,0)
    def forward(self, input, state):
        #input shape should be (number,feature size)
        values_i = F.linear(input, self.weights_i,self.input_bias)
        pre_gate_ih = F.linear(input, self.gate_weights_ih)
        pre_gate = pre_gate_ih + self.gate_bias
        gate = self.gate_function(pre_gate,training=self.training)
        pre_h = state*gate+ values_i
        new_h = self.recurrent_function(pre_h,self.training)
        return new_h,pre_gate,gate,values_i,pre_h,pre_gate_ih

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
            return (freq.transpose(0,1)*inform).transpose(0,1)#inform*freq
        elif len(x.shape)==2:
            return (freq.transpose(0,1)*inform).transpose(0,1)
        else:
            raise Exception("Shape is not permmited.")
            
def noisy_relu_half_max_float32(alpha=1.0, c=0.05, p=1):
    half_relu_max_float32 = Hardtanh(0,np.finfo(np.float32).max/2)
    return noisy_hard_function(half_relu_max_float32,alpha,c,p)

class ConcatCNN(nn.Module):
    def __init__(self,in_channels,layer_num,kernel_sizes,outputs,ln_mode=None,
                 with_pwm=True):
        super().__init__()
        self.in_channels = in_channels
        self.layer_num = layer_num
        self.kernel_sizes = kernel_sizes
        self.outputs = outputs
        self.with_pwm = with_pwm
        if ln_mode in ["before_cnn","after_cnn","after_activation",None]:
            self.ln_mode = ln_mode
        else:
            raise Exception('ln_mode should be "before_cnn", "after_cnn", None, or "after_activation"')
        self.activation_function = noisy_relu()
        self._build_layers()
        self.reset_parameters()
    def _build_layers(self):
        in_num = self.in_channels
        self.lns = []
        self.cnns = []
        self.pwm = PWM()
        for index in range(self.layer_num):
            kernel_size = self.kernel_sizes[index]
            output = self.outputs[index]
            cnn = nn.Conv1d(in_channels=in_num,kernel_size=kernel_size,
                            out_channels=output)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            if self.ln_mode is not None:
                in_channel = {"before_cnn":in_num,"after_cnn":output,"after_activation":output}
                ln = nn.LayerNorm(in_channel[self.ln_mode])
                self.lns.append(ln)
                setattr(self, 'ln_'+str(index), ln)
            in_num += output
        self.out_channels = in_num
    def reset_parameters(self):
        for layer in self.children():
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
            x_ = cnn(x_)
            distribution_output['cnn_x_'+str(index)] = x_
            if self.ln_mode=='after_cnn':
                ln = self.lns[index]
                x_ = self._layer_normalized(ln,x_,'ln_'+str(index),distribution_output)
                if self.with_pwm:
                    x_ = self.pwm(x_)
            x_ = self.activation_function(x_,self.training)
            distribution_output['post_act_x_'+str(index)] = x_
            if self.ln_mode=='after_activation':
                ln = self.lns[index]
                x_ = self._layer_normalized(ln,x_,'ln_'+str(index),distribution_output)
                if self.with_pwm:
                    x_ = self.pwm(x_)
            x_,lengths = Concat(dim=1)([pre_x,x_],lengths)
        distribution_output['cnn_result'] = x_
        return x_,lengths,distribution_output

class SeqAnnlLoss(nn.Module):
    def __init__(self, class_num,alphas=None, gamma=0,ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self._ignore_index = ignore_index
        self._class_num = class_num
        self._alphas = alphas
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
        if self._ignore_index is not None:
            mask = (target.max(1)[0] != self._ignore_index).float()
        decay_cooef = (1-pt)**self.gamma
        loss_ =  -decay_cooef* (pt+1e-32).log() * target
        if spatial_weights is not None:
            if list(pt.shape) != list(spatial_weights.shape):
                 spatial_weights = spatial_weights.transpose(0,2)[:input_length].transpose(0,2)
            spatial_weights = torch.FloatTensor(spatial_weights)#.cuda()
        if self._alphas is not None:
            loss_ = (loss_.transpose(1,2)*self._alphas).transpose(1,2)
        loss_ = loss_.sum(1)
        if self._ignore_index is not None:
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
    
class ConcatRNN(nn.Module):
    def __init__(self,in_channels,layer_num,outputs,rnn_cell_class,rnns_as_output=True,
                 layer_norm=False,bidirectional=True,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.layer_num = layer_num
        self.outputs = outputs
        self.layer_norm = layer_norm
        self.kwargs = kwargs
        self.rnn_cell_class = rnn_cell_class
        self.rnns_as_output = rnns_as_output
        self.bidirectional = bidirectional
        self._build_layers()
        self.reset_parameters()
    def _build_layers(self):
        in_num = self.in_channels
        self.rnns = []
        self.lns=[]
        for index in range(self.layer_num):
            output_num = self.outputs[index]
            rnn_cell = self.rnn_cell_class(input_size=in_num,hidden_size=output_num)
            rnn = RNN(rnn_cell,bidirectional=self.bidirectional,**self.kwargs)
            self.rnns.append(rnn)
            setattr(self, 'rnn_'+str(index), rnn)
            if self.layer_norm:
                ln = nn.LayerNorm(output_num*2)
                self.lns.append(ln)
                setattr(self, 'ln_'+str(index), ln)
            in_num += output_num*2
        if self.rnns_as_output:
            self.out_channels = output_num*2*self.layer_num
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
        for index in range(self.layer_num):
            rnn = self.rnns[index]
            pre_x = x_
            x_ = rnn(x_,lengths)
            if hasattr(rnn,'output_names'):
                for name,item in zip(rnn.output_names,x_):
                    if name not in rnn_output.keys():
                        rnn_output[name] = []
                    rnn_output[name].append(item)
                    values = item.transpose(1,2)
                    distribution_output["rnn_"+str(index)+"_"+name] = values
                x_ = x_[rnn.output_names.index('new_h')]
            else:
                if 'new_h' not in rnn_output.keys():
                    rnn_output['new_h'] = []
                rnn_output['new_h'].append(x_)
            if self.layer_norm:
                x_ = self.lns[index](x_)
            x_ = Concat(dim=1)([pre_x.transpose(1,2),x_.transpose(1,2)])
            x_ = x_.transpose(1,2)
        for name,value in rnn_output.items():
            temp = torch.cat(value,2)
            rnn_output[name] = temp.transpose(1, 2)
        if self.rnns_as_output:
            x = rnn_output['new_h']
        else:
            x = x_.transpose(1, 2)
        distribution_output['rnn_result'] = rnn_output['new_h']
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
    def __init__(self,in_channels,out_channels,rnn_cell_class,
                 cnn_num=0,cnn_kernel_sizes=128,cnn_outputs=[],
                 rnn_num=0,rnn_outputs=[],reduce_cnn_ratio=1,
                 init_value=0,with_pwm=True,
                 use_CRF=False,rnn_layer_norm=False,**kwargs):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels = out_channels
        self.cnn_num=cnn_num
        self.cnn_kernel_sizes=cnn_kernel_sizes
        self.cnn_outputs=cnn_outputs
        self.rnn_num=rnn_num
        self.rnn_outputs = rnn_outputs
        self.rnn_cell_class=rnn_cell_class
        self.rnn_layer_norm = rnn_layer_norm
        self.reduce_cnn_ratio = reduce_cnn_ratio
        self.init_value = init_value
        self.use_CRF = use_CRF
        self.with_pwm = with_pwm
        self.CRF = None
        self._saved_index_outputs = None
        self._outputs = None
        self._lengths = None
        self._distribution = None
        self._build_layers()
        self.reset_parameters()
    def _build_layers(self):
        in_channels=self.in_channels
        if self.cnn_num > 0:
            self.cnns = ConcatCNN(in_channels,self.cnn_num,self.cnn_kernel_sizes,
                                  self.cnn_outputs,ln_mode="before_cnn",with_pwm=self.with_pwm)
            self.cnn_ln = nn.LayerNorm(self.cnns.out_channels)
            self.cnn_pwm = PWM()
            if self.reduce_cnn_ratio<1:
                self.cnn_merge = nn.Conv1d(in_channels=self.cnns.out_channels,kernel_size=1,
                                           out_channels=int(self.cnns.out_channels*self.reduce_cnn_ratio))
            in_channels=int(self.cnns.out_channels*self.reduce_cnn_ratio)
        if self.rnn_num > 0:
            self.rnns =  ConcatRNN(in_channels,self.rnn_num,
                                   self.rnn_outputs,self.rnn_cell_class,
                                   init_value=self.init_value,
                                   layer_norm=self.rnn_layer_norm)
            in_channels=self.rnns.out_channels
        self.last = nn.Conv1d(in_channels=in_channels,kernel_size=1,
                              out_channels=self.out_channels)
        self.CRF = BatchCRF(self.out_channels)
       
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer,'reset_parameters'):
                layer.reset_parameters()
        if self.cnn_num > 0 and self.reduce_cnn_ratio<1:
            normal_(self.cnn_merge.weight)
            constant_(self.cnn_merge.bias,0)
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
            x = self.cnn_pwm(x)
            if self.reduce_cnn_ratio < 1:
                x = self.cnn_merge(x)
            distribution_output.update(cnn_distribution)
            print("CNN time",int(time.time()-pre_time))
        if self.rnn_num > 0:
            pre_time = time.time()
            distribution_output["pre_rnn_result"] = x
            x,lengths,rnn_distribution = self.rnns(x,lengths)
            distribution_output["pre_last_result"] = x
            distribution_output.update(rnn_distribution)
            print("RNN time",int(time.time()-pre_time))
        x = self.last(x)
        if not self.use_CRF:
            distribution_output["pre_softmax"] = x
            x = nn.LogSoftmax(dim=1)(x)
            distribution_output["log_softmax"] = x
            x = x.exp()
            distribution_output["softmax"] = x
        self._distribution = distribution_output
        self._lengths = lengths
        self._outputs = x
        if self.use_CRF is None:
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