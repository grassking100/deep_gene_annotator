import torch
from torch import nn
from torch.nn.init import zeros_
from torch.nn import ReLU
from .noisy_activation import NoisyReLU
from .customized_layer import Conv1d, Concat,PaddedBatchNorm1d, BasicModel, add
from .utils import xavier_uniform_in_

ACTIVATION_FUNC = {'NoisyReLU':NoisyReLU(),'ReLU':ReLU()}

def customized_init_cnn(cnn,gain=None):
    relu_gain = nn.init.calculate_gain('relu')
    gain = relu_gain if gain is None else gain
    xavier_uniform_in_(cnn.weight,gain)
    zeros_(cnn.bias)

class CANBlock(BasicModel):
    def __init__(self,in_channels,kernel_size,out_channels,
                 norm_mode=None,norm_type=None,
                 activation_function=None,customized_init=None,**kwargs):
        super().__init__()
        self.name = ""
        self.customized_init = customized_init
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
                          out_channels=self.out_channels,bias=use_bias,**kwargs)
        if self.norm_mode is not None:
            in_channel = {"before_cnn":self.in_channels,"after_cnn":self.out_channels,
                          "after_activation":self.out_channels}
            if self.norm_type is None:
                self.norm = PaddedBatchNorm1d(in_channel[self.norm_mode])
            else:
                self.norm = self.norm_type(in_channel[self.norm_mode])

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
        return config
    
    def reset_parameters(self):
        super().reset_parameters()
        if self.customized_init is not None:
            self.customized_init(self.cnn)

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
                                       norm_mode=norm_mode)
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
