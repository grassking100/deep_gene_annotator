import torch
from .customized_layer import BasicModel
from .cnn import STACK_CNN_CLASSES, CANBlock


class FeatureSiteBlock(BasicModel):
    def __init__(self,feature_block,site_block):
        super().__init__()
        self._basic_feature_block = feature_block
        self._site_block = site_block
        self._in_channels = basic_feature_block.in_channels
        self._out_channels = basic_feature_block.out_channels + site_block.out_channels
        self._concat = Concat(dim=1)

    def get_config(self):
        config = super().get_config()
        config['basic_feature_block'] = self._basic_feature_block.get_config()
        config['site_block'] = self._site_block.get_config()
        return config

    def forward(self, x, lengths, **kwargs):
        features, lengths = self._basic_feature_block(x, lengths, **kwargs)
        predicted_sites = self._site_block(features, lengths)[0]
        results = self._concat(features,predicted_sites)
        return results, lengths
    

class FeatureBlockBuilder:
    IN_CHANNELS = 4
    HIDDEN_SIZE = 4
    KERNEL_SIZE = 16
    NUM_LAYERS = 16
    STACK_TYPE = 'ConcatCNN'
    KWARGS = {}
    SITE_NUM = None
    def __init__(self):
        self._in_channels = self._out_channels = self._kernel_size = None
        self._num_layers = self._stack_type = self._kwargs = None
        self._site_num = None
        self.reset()
        
    def set_basic_settings(self,in_channels=None,hidden_size=None,kernel_size=None,
                           num_layers=None,stack_type=None,**kwargs):
        self._in_channels = in_channels or self.IN_CHANNELS
        self._hidden_size = hidden_size or self.HIDDEN_SIZE
        self._kernel_size = kernel_size or self.KERNEL_SIZE
        self._num_layers = num_layers or self.NUM_LAYERS
        self._stack_type = stack_type or self.STACK_TYPE
        self._kwargs = kwargs or dict(self.KWARGS)
        
    def reset(self):
        self._in_channels = self.IN_CHANNELS
        self._hidden_size = self.HIDDEN_SIZE
        self._kernel_size = self.KERNEL_SIZE
        self._num_layers = self.NUM_LAYERS
        self._stack_type = self.STACK_TYPE
        self._kwargs = dict(self.KWARGS)
        self._site_num = self.SITE_NUM
        
    def add_site_block_settings(self,site_num):
        self._site_num = site_num
        
    def build(self):
        stack_class = STACK_CNN_CLASSES[self._stack_type]
        feature_block = stack_class(self._in_channels,self._hidden_size,self._kernel_size,
                                    self._num_layers,**self._kwargs)
        
        if self._site_num is not None:
            site_kwargs = dict(sefl._kwargs)
            site_kwargs['act_func'] = torch.softmax
            site_block = CANBlock(feature_block.out_channels,self._site_num,1,**site_kwargs)
            feature_block = FeatureSiteBlock(feature_block,site_block)
            
        return feature_block
