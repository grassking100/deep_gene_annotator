import torch.nn.functional as F
from sequence_annotation.process.utils import get_seq_mask
import torch
from torch import nn
from torch.nn.init import zeros_
from torch.nn import ReLU
from .customized_layer import Concat, BasicModel,generate_norm_class
from .utils import xavier_uniform_extend_
from .noisy_activation import SimpleHalfNoisyReLU,NoisyReLU,SimpleNoisyReLU

ACTIVATION_FUNC = {'ReLU': ReLU(),'SimpleHalfNoisyReLU':SimpleHalfNoisyReLU(),
                   'NoisyReLU':NoisyReLU(),'SimpleNoisyReLU':SimpleNoisyReLU()}
EPSILON = 1e-32


def kaiming_uniform_cnn_init(cnn, mode=None):
    mode = mode or 'fan_in'
    gain = nn.init.calculate_gain('relu')
    xavier_uniform_extend_(cnn.weight, gain, mode)
    if cnn.bias is not None:
        zeros_(cnn.bias)


def xavier_uniform_cnn_init(cnn, mode=None):
    mode = mode or 'fan_in'
    gain = nn.init.calculate_gain('linear')
    xavier_uniform_extend_(cnn.weight, gain, mode)
    if cnn.bias is not None:
        zeros_(cnn.bias)


CNN_INIT_MODES = {
    None: None,
    'kaiming_uniform_cnn_init': kaiming_uniform_cnn_init,
    'xavier_uniform_cnn_init': xavier_uniform_cnn_init
}


def create_mask(x, lengths):
    mask_ = get_seq_mask(lengths)
    mask_ = mask_.to(x.dtype).unsqueeze(1)
    mask = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.dtype)
    if x.is_cuda:
        mask = mask.cuda()
        mask_ = mask_.cuda()
    mask[:, :, :max(lengths)] += mask_
    return mask


class Conv1d(BasicModel):
    def __init__(self,in_channels,out_channels,kernel_size,bias=True,padding_value=None,
                 customized_init=None):
        super().__init__()
        self._cnn = nn.Conv1d(in_channels,out_channels,kernel_size,bias=bias)
        if isinstance(customized_init, str):
            customized_init = CNN_INIT_MODES[customized_init]
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._customized_init = customized_init or CNN_INIT_MODES['xavier_uniform_cnn_init']
        self._padding_value = padding_value or 0
        self._pad_func = None
        self._bias = bias

        if kernel_size > 1:
            if kernel_size % 2 == 0:
                bound = int(kernel_size / 2)
                self._pad_func = nn.ConstantPad1d((bound - 1, bound),self._padding_value)
            else:
                bound = int((kernel_size - 1) / 2)
                self._pad_func = nn.ConstantPad1d((bound, bound),self._padding_value)

        self._mask_kernel = torch.ones((out_channels, in_channels, kernel_size)).cuda()
        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config['bias'] = self._bias
        config['kernel_size'] = self._kernel_size
        config['padding_value'] = self._padding_value
        config['customized_init'] = str(self._customized_init)
        return config

    def reset_parameters(self):
        super().reset_parameters()
        if self._customized_init is not None:
            self._customized_init(self._cnn)

    def forward(self, x, lengths=None, masks=None):
        # N,C,L
        if lengths is None:
            lengths = [x.shape[2]] * x.shape[0]
        if masks is None:
            masks = create_mask(x, lengths)
        if self._kernel_size > 1:
            x = self._pad_func(x)
        x = self._cnn.forward(x)
        if self._kernel_size > 1:
            x = masks.repeat(1, x.shape[1], 1) * x
        return x, lengths, masks


class CANBlock(BasicModel):
    def __init__(self,in_channels,out_channels,kernel_size,name=None,
                 norm_mode=None,norm_class=None,act_func=None,
                 dropout=None,dropout_mode=None,customized_init=None,
                 **kwargs):
        customized_init = customized_init or CNN_INIT_MODES['kaiming_uniform_cnn_init']
        super().__init__()
        if norm_mode not in ["before_cnn", "after_cnn", "after_activation", None]:
            raise Exception("The norm_mode should be \"before_cnn\", \"after_cnn\","
                "None, or \"after_activation\", but got {}".format(norm_mode))
        if dropout_mode not in ["in", "out",None]:
            raise Exception("The norm_mode should be \"in\" and"
                "\"out\", but got {}".format(dropout_mode))
        self._name = ""
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._norm_class = norm_class or generate_norm_class('PaddedBatchNorm1d')
        self._kwargs = kwargs
        self._norm_mode = norm_mode or 'after_activation'
        self._dropout_mode = dropout_mode or 'out'
        if act_func is None:
            self._act = ReLU()
        elif isinstance(act_func, str):
            self._act = ACTIVATION_FUNC[act_func]
        else:
            self._act = act_func
        use_bias = self._norm_mode != "after_cnn"
        self._cnn = Conv1d(in_channels,out_channels,kernel_size,bias=use_bias,
                           customized_init=customized_init,**kwargs)
        in_channel = {
            "before_cnn": self.in_channels,
            "after_cnn": self.out_channels,
            "after_activation": self.out_channels
        }
        self._norm = self._norm_class(in_channel[self._norm_mode])
        self._dropout = dropout or 0
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config.update(self._cnn.get_config())
        config['name'] = self._name
        config['norm_mode'] = self._norm_mode
        if self._norm is not None:
            config['norm_config'] = self._norm.get_config()
        config['activation'] = str(self._act)
        config['dropout'] = self._dropout
        config['dropout_mode'] = self._dropout_mode
        return config
        
    @property
    def name(self):
        return self._name
        
    def _normalized(self, x, lengths):
        x = self._norm(x, lengths)
        self._update_distribution(x, key='norm_{}'.format(self._name))
        return x

    def forward(self, x, lengths, **kwargs):
        # X shape : N,C,L
        if self._dropout > 0 and self._training and self._dropout_mode == "in":
            x = F.dropout(x, self._dropout, self.training)
            
        if self._norm_mode == 'before_cnn':
            x = self._normalized(x, lengths)
        x, lengths, masks = self._cnn(x, lengths, **kwargs)
        self._update_distribution(x, key='cnn_x_{}'.format(self._name))
        if self._norm_mode == 'after_cnn':
            x = self._normalized(x, lengths)
        x = self._act(x)
        self._update_distribution(x, key='post_act_x_{}'.format(self._name))
        if self._norm_mode == 'after_activation':
            x = self._normalized(x, lengths)

        if self._dropout > 0 and self.training and self._dropout_mode == "out":
            x = F.dropout(x, self._dropout, self.training)
        return x, lengths, masks


class StackCNN(BasicModel):
    def __init__(self,in_channels,hidden_size,kernel_size,num_layers,
                 norm_class=None,**kwargs):
        super().__init__()
        self._in_channels = in_channels
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._norm_class = norm_class
        self._cnns = []
        self._kwargs = kwargs

    def get_config(self):
        config = super().get_config()
        config.update(self._kwargs)
        config['num_layers'] = self._num_layers
        cnn_config = dict(self._cnns[0].get_config())
        del cnn_config['in_channels']
        del cnn_config['out_channels']
        del cnn_config['norm_config']
        cnn_config['hidden_size'] = self._hidden_size
        config['can_config'] = cnn_config
        if self._norm_class is not None:
            norm_config = dict(self._norm_class(self.in_channels).get_config())
            del norm_config['in_channels']
            del norm_config['out_channels']
            config['norm_class_config'] = norm_config
        return config


class ConcatCNN(StackCNN):
    def __init__(self,in_channels,hidden_size,kernel_size,num_layers,
                 norm_class=None,**kwargs):
        super().__init__(in_channels,hidden_size,kernel_size,num_layers,
                         norm_class=norm_class,**kwargs)
        in_num = in_channels
        for index in range(self._num_layers):
            setting = {}
            for key, value in kwargs.items():
                if isinstance(value, list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,hidden_size,kernel_size,name=str(index),
                           norm_class=self._norm_class,**setting)
            self._cnns.append(cnn)
            setattr(self, 'cnn_' + str(index), cnn)
            in_num = in_num + cnn.out_channels
        self._out_channels = in_num
        self._concat = Concat(dim=1)
        self.reset_parameters()

    def forward(self, x, lengths,**kwargs):
        # X shape : N,C,L
        masks = None
        for index in range(self._num_layers):
            pre_x = x
            cnn = self._cnns[index]
            x, lengths, masks = cnn(x,lengths=lengths,masks=masks,**kwargs)
            self._update_distribution(cnn.saved_distribution)
            x = self._concat([pre_x, x])
        self._update_distribution(x, key='cnn_result')
        return x, lengths


class ResCNN(StackCNN):
    def __init__(self,in_channels,hidden_size,kernel_size,num_layers,
                 norm_class=None,**kwargs):
        super().__init__(in_channels,hidden_size,kernel_size,num_layers,
                         norm_class=norm_class,**kwargs)
        in_num = in_channels
        for index in range(self._num_layers):
            setting = {}
            for key, value in kwargs.items():
                if isinstance(value, list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,hidden_size,kernel_size,
                           norm_class=self._norm_class,
                           name = str(index),**setting)
            self._cnns.append(cnn)
            setattr(self, 'cnn_' + str(index), cnn)
            in_num = cnn.out_channels
        self._out_channels = in_num
        self.reset_parameters()

    def forward(self, x, lengths,**kwargs):
        # X shape : N,C,L
        masks = None
        for index in range(self._num_layers):
            pre_x = x
            cnn = self._cnns[index]
            x, lengths, masks = cnn(x,lengths=lengths,masks=masks,**kwargs)
            self._distribution.update(cnn.saved_distribution)
            if index > 0:
                x = pre_x + x
        self._update_distribution(x, key='cnn_result')
        return x, lengths

STACK_CNN_CLASSES = {'ConcatCNN': ConcatCNN, 'ResCNN': ResCNN}
