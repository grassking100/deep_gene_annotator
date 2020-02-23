import warnings
import math
import torch.nn.functional as F
from sequence_annotation.process.utils import get_seq_mask
import torch
from torch import nn
from torch.nn.init import zeros_
from torch.nn import ReLU
from .noisy_activation import NoisyReLU
from .customized_layer import Concat, BasicModel, Add, generate_norm_class
from .utils import xavier_uniform_extend_

ACTIVATION_FUNC = {'NoisyReLU':NoisyReLU(),'ReLU':ReLU()}
PADDING_HANDLE = ['valid','same','partial']
EPSILON = 1e-32
    
def kaiming_uniform_cnn_init(cnn,mode=None):
    mode = mode or 'fan_in'
    gain = nn.init.calculate_gain('relu')
    xavier_uniform_extend_(cnn.weight,gain,mode)
    if cnn.bias is not None:
        zeros_(cnn.bias)
    
def xavier_uniform_cnn_init(cnn,mode=None):
    mode = mode or 'fan_in'
    gain = nn.init.calculate_gain('linear')
    xavier_uniform_extend_(cnn.weight,gain,mode)
    if cnn.bias is not None:
        zeros_(cnn.bias)

CNN_INIT_MODE = {None:None,'kaiming_uniform_cnn_init':kaiming_uniform_cnn_init,'xavier_uniform_cnn_init':xavier_uniform_cnn_init}

def create_mask(x,lengths):
    mask = torch.zeros(x.shape[0],1,x.shape[2]).to(x.dtype)
    if x.is_cuda:
        mask = mask.cuda()
    mask_ = get_seq_mask(lengths,to_cuda=x.is_cuda).to(x.dtype).unsqueeze(1)
    mask[:,:,:max(lengths)] += mask_
    return mask

class Conv1d(nn.Conv1d,BasicModel):
    def __init__(self,padding_handle=None,padding_value=None,customized_init=None,
                 *args,**kwargs):
        if isinstance(customized_init,str):
            customized_init = CNN_INIT_MODE[customized_init]
        self.customized_init = customized_init
        super().__init__(*args,**kwargs)
        self.args = args
        self.kwargs = kwargs
        padding_handle = padding_handle or 'valid'
        self.padding_value = padding_value or 0
        self._pad_func = None
        kernek_size = self.kernel_size[0]
        self.full_size = self.in_channels*kernek_size
        if padding_handle in PADDING_HANDLE:
            self._padding_handle = padding_handle
        else:
            raise Exception("Invalid mode {} to handle padding".format(padding_handle))
        if self._padding_handle != 'valid':
            if self.padding != (0,) or self.dilation != (1,) or self.stride != (1,):
                raise Exception("The padding_handle sholud be valid to set padding, dilation and stride at the same time")
            
        if kernek_size>1:
            if self._padding_handle == 'partial' and self.padding_value != 0:
                 raise Exception("When padding_handle is partial, the padding_value should be 0")
            if kernek_size%2 == 0:
                bound = int(kernek_size/2)
                self._pad_func = nn.ConstantPad1d((bound-1,bound),self.padding_value)
            else:
                bound = int((kernek_size-1)/2)
                self._pad_func = nn.ConstantPad1d((bound,bound),self.padding_value)
                
        self.mask_kernel = torch.ones((self.out_channels,self.in_channels,self.kernel_size[0])).cuda()
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config['args'] = self.args
        config['kwargs'] = self.kwargs
        config['padding_handle'] = self._padding_handle
        config['padding_value'] = self.padding_value
        config['customized_init'] = str(self.customized_init)
        return config
        
    def reset_parameters(self):
        super().reset_parameters()
        if self.customized_init is not None:
            self.customized_init(self)
            
    def forward(self,x,lengths=None,weights=None,mask=None):
        #N,C,L
        if lengths is None:
            lengths = [x.shape[2]]*x.shape[0]
        #if self.kernel_size[0] > 1:
        if mask is None:
            mask = create_mask(x,lengths)
        if self._padding_handle == 'same' or self.kernel_size[0] == 1:
            if self.kernel_size[0] > 1:
                x = self._pad_func(x)
            x = super().forward(x)
            if self.kernel_size[0] > 1:
                x = mask.repeat(1,x.shape[1],1) * x
            new_lengths = lengths
        elif self._padding_handle == 'valid':
            x = super().forward(x)
            padding = sum(self.padding)
            dilation= sum(self.dilation)
            stride= sum(self.stride)
            change = 2*padding-dilation*(self.kernel_size[0]-1)-1
            new_lengths = [math.floor((length + change)/stride) + 1 for length in lengths]
            mask = create_mask(x,new_lengths)
            x = mask[:,:x.shape[2]].repeat(1,x.shape[1],1) * x
            x = x[:,:,:max(new_lengths)]
        elif self._padding_handle == 'partial':
            new_lengths = lengths
            if weights is None:
                warnings.warn("Caution: weights can be reused ONLY if kernel sizes of previous layer "+
                              "and current layer is the same, please check this by yourself")
                padded_mask = self._pad_func(mask)
                if x.is_cuda:
                    padded_mask = padded_mask.cuda()
                mask_sum = F.conv1d(padded_mask, self.mask_kernel, bias=None)
                weights = self.full_size /(mask_sum+ EPSILON)
                if x.is_cuda:
                    weights = weights.cuda()
            x = self._pad_func(x)
            x = F.conv1d(x, self.weight, bias=None)
            x = torch.mul(x,weights)
            if self.bias is not None:
                x = x+self.bias.view(1, self.out_channels, 1)
            x = mask.repeat(1,x.shape[1],1) * x                
        else:
            new_lengths = lengths
        return x, new_lengths, weights, mask

class CANBlock(BasicModel):
    def __init__(self,in_channels,norm_mode=None,norm_type=None,
                 activation_function=None,dropout=None,**kwargs):
        super().__init__()
        self.name = ""
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.kwargs = kwargs
        if norm_mode in ["before_cnn","after_cnn","after_activation",None]:
            self.norm_mode = norm_mode
        else:
            raise Exception("The norm_mode should be \"before_cnn\", \"after_cnn\","
                            "None, or \"after_activation\", but got {}".format(norm_mode))
        if activation_function is None:
            self.activation_function = NoisyReLU()
        elif isinstance(activation_function,str):
            self.activation_function = ACTIVATION_FUNC[activation_function]
        else:
            self.activation_function = activation_function
        use_bias = self.norm_mode != "after_cnn"
        self.cnn = Conv1d(in_channels=self.in_channels,bias=use_bias,**kwargs)
        self.kernel_size = self.cnn.kernel_size
        self.out_channels = self.cnn.out_channels
        if self.norm_mode is not None:
            in_channel = {"before_cnn":self.in_channels,"after_cnn":self.out_channels,
                          "after_activation":self.out_channels}
            self.norm = self.norm_type(in_channel[self.norm_mode])
        self.dropout = dropout or 0
        self.reset_parameters()

    def _normalized(self,x,lengths):
        x = self.norm(x,lengths)
        self.update_distribution(x,key='norm_{}'.format(self.name))
        return x

    def forward(self, x, lengths,weights=None,mask=None):
        #X shape : N,C,L
        if self.norm_mode=='before_cnn':
            x = self._normalized(x,lengths)
        x,lengths,weights,mask = self.cnn(x,lengths,weights,mask)
        self.update_distribution(x,key='cnn_x_{}'.format(self.name))
        if self.norm_mode=='after_cnn':
            x = self._normalized(x,lengths)
        x = self.activation_function(x)
        self.update_distribution(x,key='post_act_x_{}'.format(self.name))
        if self.norm_mode=='after_activation':
            x = self._normalized(x,lengths)
        if self.dropout is not None and self.training:
            x = F.dropout(x,self.dropout,self.training)
        return x,lengths,weights,mask
    
    def get_config(self):
        config = super().get_config()
        config.update(self.cnn.get_config())
        config['name'] = self.name
        config['norm_type'] = str(self.norm_type)
        config['norm_mode'] = self.norm_mode
        config['activation_function'] = str(self.activation_function)
        config['dropout'] = self.dropout
        return config

class StackCNN(BasicModel):
    def __init__(self,in_channels,num_layers,norm_type=None,norm_input=False,
                 norm_momentum=None,norm_affine=True,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.norm_input = norm_input
        self.norm_affine = norm_affine
        self.norm_momentum = norm_momentum or 0.1
        self.norm_type = norm_type or 'PaddedBatchNorm1d'
        self.norm_class = generate_norm_class(self.norm_type,affine=self.norm_affine,
                                              momentum=self.norm_momentum)
        self.norm = None
        if self.norm_input:
            self.norm = self.norm_class(self.in_channels)
        self.kwargs = kwargs
    
    def handle_input(self, x, lengths):
        if self.norm_input:
            return self.norm(x,lengths)
        else:
            return x
    
    def get_config(self):
        config = super().get_config()
        config.update(self.kwargs)
        config['num_layers'] = self.num_layers
        config['norm_type'] = self.norm_type
        config['norm_input'] = self.norm_input
        config['norm_momentum'] = self.norm_momentum
        config['norm_affine'] = self.norm_affine
        
        return config
    
class ConcatCNN(StackCNN):
    def __init__(self,in_channels,num_layers,bottleneck_factor=None,norm_type=None,
                 norm_input=False,norm_momentum=None,norm_affine=True,**kwargs):
        super().__init__(in_channels,num_layers,norm_type=norm_type,norm_input=norm_input,
                         norm_momentum=norm_momentum,norm_affine=norm_affine,**kwargs)
        self.bottleneck_factor = bottleneck_factor or 0
        in_num = in_channels
        self.cnns = []
        self.bottlenecks = []
        self.length_change = True
        if 'padding_handle' in self.kwargs:
            if self.kwargs['padding_handle'] in ['partial','same']:
                self.length_change = False
        for index in range(self.num_layers):
            in_num_ = in_num
            if self.bottleneck_factor > 0:
                out_num = int(in_num*self.bottleneck_factor)
                out_num = 1 if out_num == 0 else out_num
                bottleneck = CANBlock(in_channels=in_num,kernel_size=1,
                                       out_channels=out_num,
                                       norm_type=self.norm_class,**kwargs)
                self.bottlenecks.append(bottleneck)
                setattr(self, 'bottleneck_'+str(index), bottleneck)
                in_num = out_num
            setting = {}
            for key,value in kwargs.items():
                if isinstance(value,list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,norm_type=self.norm_class,**setting)
            cnn.name=str(index)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            in_num = in_num_ + cnn.out_channels
        self.out_channels = in_num
        self.concat = Concat(handle_length=self.length_change,dim=1)
        self.reset_parameters()

    def forward(self, x, lengths,weights=None,mask=None):
        #X shape : N,C,L
        x = self.handle_input(x,lengths)
        for index in range(self.num_layers):
            pre_x = x
            cnn = self.cnns[index]
            if self.bottleneck_factor > 0:
                bottleneck = self.bottlenecks[index]
                x,lengths,_ = bottleneck(x,lengths)
            x,lengths,weights,mask = cnn(x,lengths=lengths,weights=weights)
            self.update_distribution(cnn.saved_distribution)
            x = self.concat([pre_x,x])
        self.update_distribution(x,key='cnn_result')
        return x,lengths,weights
    
    def get_config(self):
        config = super().get_config()
        config['bottleneck_factor'] = self.bottleneck_factor
        return config
    
class ResCNN(StackCNN):
    def __init__(self,in_channels,num_layers,norm_type=None,norm_affine=True,
                 norm_input=False,norm_momentum=None,bottleneck_factor=None,**kwargs):
        super().__init__(in_channels,num_layers,norm_type=norm_type,norm_affine=norm_affine,
                         norm_input=norm_input,norm_momentum=norm_momentum,**kwargs)
        if bottleneck_factor is not None:
            raise Exception("The bottleneck has not be implemented in ResCNN".format(bottleneck_factor))
        in_num = in_channels
        self.length_change = True
        if 'padding_handle' in self.kwargs:
            if self.kwargs['padding_handle'] in ['partial','same']:
                self.length_change = False
                
        self.cnns = []
        for index in range(self.num_layers):
            setting = {}
            for key,value in kwargs.items():
                if isinstance(value,list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,norm_type=self.norm_class,**setting)
            cnn.name=str(index)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            in_num = cnn.out_channels
        self.out_channels = in_num
        self.reset_parameters()
        self.add = Add(handle_length=self.length_change)

    def forward(self, x, lengths,weights=None,mask=None):
        #X shape : N,C,L
        x = self.handle_input(x,lengths)
        for index in range(self.num_layers):
            pre_x = x
            cnn = self.cnns[index]
            x,lengths,weights,mask = cnn(x,lengths=lengths,weights=weights,mask=mask)
            self._distribution.update(cnn.saved_distribution)
            if index > 0:
                x = self.add(pre_x,x)
        self.update_distribution(x,key='cnn_result')
        return x,lengths,weights
    
STACK_CNN_CLASS = {'ConcatCNN':ConcatCNN,'ResCNN':ResCNN}
