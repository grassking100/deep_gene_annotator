import warnings
import math
import torch.nn.functional as F
from sequence_annotation.process.utils import get_seq_mask
import torch
from torch import nn
from torch.nn.init import zeros_,xavier_uniform_
from torch.nn import ReLU
from .noisy_activation import NoisyReLU
from .customized_layer import Concat,PaddedBatchNorm1d, BasicModel, add
from .utils import xavier_uniform_extend_

ACTIVATION_FUNC = {'NoisyReLU':NoisyReLU(),'ReLU':ReLU()}
PADDING_HANDLE = ['valid','same','partial']
EPSILON = 1e-32
    
def kaming_uniform_cnn_init(cnn,mode=None):
    mode = mode or 'fan_in'
    gain = nn.init.calculate_gain('relu')
    xavier_uniform_extend_(cnn.weight,gain,mode)
    zeros_(cnn.bias)

CNN_INIT_MODE = {None:None,'kaming_uniform_cnn_init':kaming_uniform_cnn_init}

class Conv1d(nn.Conv1d):
    def __init__(self,padding_handle=None,padding_value=None,customized_init=None,
                 *args,**kwargs):
        if isinstance(customized_init,str):
            customized_init = CNN_INIT_MODE[customized_init]
        self.customized_init = customized_init
        super().__init__(*args,**kwargs)
        padding_handle = padding_handle or 'valid'
        padding_value = padding_value or 0
        self._pad_func = None
        self.mask_kernel = None
        kernek_size = self.kernel_size[0]
        self.full_size = self.in_channels*kernek_size
        if padding_handle in PADDING_HANDLE:
            self._padding_handle = padding_handle
        else:
            raise Exception("Invalid mode {} to handle padding".format(padding_handle))
        if self._padding_handle != 'valid':
            if self.padding != (0,) or self.dilation != (1,) or self.stride != (1,):
                raise Exception("The padding_handle sholud be valid to set padding, dilation and stride at the same time")
            
            if self._padding_handle in ['same','partial'] and kernek_size>1:
                if self._padding_handle == 'partial' and padding_value != 0:
                     raise Exception("When padding_handle is partial, the padding_value should be 0")
                if kernek_size%2 == 0:
                    bound = int(kernek_size/2)
                    self._pad_func = nn.ConstantPad1d((bound-1,bound),padding_value)
                else:
                    bound = int((kernek_size-1)/2)
                    self._pad_func = nn.ConstantPad1d((bound,bound),padding_value)

        if self._padding_handle == 'partial' and kernek_size > 1:
            self.mask_kernel = torch.ones((self.out_channels,self.in_channels,self.kernel_size[0]))
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.customized_init is not None:
            self.customized_init(self)
            
    def forward(self,x,lengths=None,weights=None):
        #N,C,L
        origin_shape = x.shape
        if lengths is None:
            lengths = [x.shape[2]]*x.shape[0]
        if self._pad_func is not None:
            x = self._pad_func(x)
        x = super().forward(x)
        if self._padding_handle == 'same':
            new_lengths = lengths
        elif self._padding_handle == 'valid':
            padding = sum(self.padding)
            dilation= sum(self.dilation)
            stride= sum(self.stride)
            change = 2*padding-dilation*(self.kernel_size[0]-1)-1
            new_lengths = [math.floor((length + change)/stride) + 1 for length in lengths]
            x = x[:,:,:max(new_lengths)]
        elif self._padding_handle == 'partial' and self.kernel_size[0] > 1:
            new_lengths = lengths
            if weights is None:
                warnings.warn("Caution: weights can be reused ONLY if kernel sizes of previous layer "+
                              "and current layer is the same, please check this by yourself")
                mask = torch.zeros(*origin_shape).to(x.dtype)
                mask_kernel = self.mask_kernel.to(x.dtype)
                if x.is_cuda:
                    mask = mask.cuda()
                    mask_kernel = mask_kernel.cuda()
                mask_ = get_seq_mask(lengths,to_cuda=x.is_cuda).unsqueeze(1).repeat(1,self.in_channels,1).to(x.dtype)
                mask[:,:,:max(lengths)] += mask_
                mask = self._pad_func(mask)
                
                mask_sum = F.conv1d(mask, mask_kernel, bias=None)
                weights = self.full_size /(mask_sum+ EPSILON)
            weights = weights[:,:,:x.shape[2]]
            if self.bias is not None:
                bias = self.bias.view(1, self.out_channels, 1)
                x = torch.mul(x-bias,weights)+bias
            else:
                x = torch.mul(x,weights)
        else:
            new_lengths = lengths
        return x, new_lengths, weights

class CANBlock(BasicModel):
    def __init__(self,in_channels,kernel_size,out_channels,
                 norm_mode=None,norm_type=None,
                 activation_function=None,customized_init=None,**kwargs):
        super().__init__()
        self.name = ""
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.kwargs = kwargs
        if norm_mode in ["before_cnn","after_cnn","after_activation",None]:
            self.norm_mode = norm_mode
        else:
            raise Exception('The norm_mode should be "before_cnn", "after_cnn", None, or "after_activation", but got {}'.format(norm_mode))
        if activation_function is None:
            self.activation_function = NoisyReLU()
        else:
            if isinstance(activation_function,str):
                self.activation_function = ACTIVATION_FUNC[activation_function]
            else:
                self.activation_function = activation_function
        use_bias = self.norm_mode != "after_cnn"
        self.cnn = Conv1d(in_channels=self.in_channels,kernel_size=self.kernel_size,
                          out_channels=self.out_channels,bias=use_bias,
                          customized_init=customized_init,**kwargs)
        if self.norm_mode is not None:
            in_channel = {"before_cnn":self.in_channels,"after_cnn":self.out_channels,
                          "after_activation":self.out_channels}
            if self.norm_type is None:
                self.norm = PaddedBatchNorm1d(in_channel[self.norm_mode])
            else:
                self.norm = self.norm_type(in_channel[self.norm_mode])
        self.customized_init = customized_init
        self.reset_parameters()

    def _normalized(self,x,lengths):
        x = self.norm(x,lengths)
        self._distribution['norm_{}'.format(self.name)] = x
        return x

    def forward(self, x, lengths,weights=None):
        #X shape : N,C,L
        if self.norm_mode=='before_cnn':
            x = self._normalized(x,lengths)
        x,lengths,weights = self.cnn(x,lengths,weights)
        self._distribution['cnn_x_{}'.format(self.name)] = x
        if self.norm_mode=='after_cnn':
            x = self._normalized(x,lengths)
        x = self.activation_function(x)
        self._distribution['post_act_x_{}'.format(self.name)] = x
        if self.norm_mode=='after_activation':
            x = self._normalized(x,lengths)
        return x,lengths,weights
    
    def get_config(self):
        config = dict(self.kwargs)
        config['name'] = self.name
        config['in_channels'] = self.in_channels
        config['kernel_size'] = self.kernel_size
        config['out_channels'] = self.out_channels
        config['norm_type'] = self.norm_type
        config['norm_mode'] = self.norm_mode
        config['activation_function'] = str(self.activation_function)
        config['customized_init'] = self.customized_init
        return config

class StackCNN(BasicModel):
    def __init__(self,in_channels,num_layers,cnn_setting,norm_mode=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.cnn_setting = cnn_setting
        self.norm_mode = norm_mode
    
    def get_config(self):
        config = {'cnn_setting':self.cnn_setting,
                  'num_layers':self.num_layers,
                  'in_channels':self.in_channels,
                  'out_channels':self.out_channels,
                  'norm_mode':self.norm_mode}
        return config
    
class ConcatCNN(StackCNN):
    def __init__(self,in_channels,num_layers,cnn_setting,norm_mode=None,
                 bottleneck_factor=None):
        super().__init__(in_channels,num_layers,cnn_setting,norm_mode)
        self.bottleneck_factor = bottleneck_factor or 0
        customized_init = None
        if 'customized_init' in cnn_setting:
            customized_init = cnn_setting['customized_init']
        in_num = in_channels
        self.cnns = []
        self.bottlenecks = []
        for index in range(self.num_layers):
            in_num_ = in_num
            if self.bottleneck_factor > 0:
                out_num = int(in_num*self.bottleneck_factor)
                out_num = 1 if out_num == 0 else out_num
                bottleneck = CANBlock(in_channels=in_num,kernel_size=1,
                                       out_channels=out_num,
                                       norm_mode=norm_mode,
                                       customized_init=customized_init)
                self.bottlenecks.append(bottleneck)
                setattr(self, 'bottleneck_'+str(index), bottleneck)
                in_num = out_num
            setting = {}
            for key,value in cnn_setting.items():
                if isinstance(value,list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,norm_mode=norm_mode,**setting)
            cnn.name=str(index)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            in_num = in_num_ + cnn.out_channels
        self.out_channels = in_num
        self.concat = Concat(dim=1)
        self.reset_parameters()

    def forward(self, x, lengths,weights=None):
        #X shape : N,C,L
        for index in range(self.num_layers):
            pre_x = x
            cnn = self.cnns[index]
            if self.bottleneck_factor > 0:
                bottleneck = self.bottlenecks[index]
                x,lengths,_ = bottleneck(x,lengths)
            x,lengths,weights = cnn(x,lengths,weights)
            self._distribution.update(cnn.saved_distribution)
            x,lengths = self.concat([pre_x,x],lengths)
        self._distribution['cnn_result'] = x
        return x,lengths,weights
    
    def get_config(self):
        config = super().get_config()
        config['bottleneck_factor'] = self.bottleneck_factor
        return config
    
class ResCNN(StackCNN):
    def __init__(self,in_channels,num_layers,cnn_setting,norm_mode=None):
        super().__init__(in_channels,num_layers,cnn_setting,norm_mode=norm_mode)
        in_num = in_channels
        self.cnns = []
        for index in range(self.num_layers):
            setting = {}
            for key,value in cnn_setting.items():
                if isinstance(value,list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,norm_mode=norm_mode,**setting)
            cnn.name=str(index)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            in_num = cnn.out_channels
        self.out_channels = in_num
        self.reset_parameters()

    def forward(self, x, lengths,weights=None):
        #X shape : N,C,L
        for index in range(self.num_layers):
            pre_x = x
            cnn = self.cnns[index]
            x,lengths,weights = cnn(x,lengths,weights)
            self._distribution.update(cnn.saved_distribution)
            if index > 0:
                x,lengths = add(pre_x,x,lengths)
        self._distribution['cnn_result'] = x
        return x,lengths,weights
    
STACK_CNN_CLASS = {'ConcatCNN':ConcatCNN,'ResCNN':ResCNN}
