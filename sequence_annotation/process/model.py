import torch
from torch import nn
from .customized_layer import BasicModel
from .rnn import GRU, LSTM,GRU_INIT_MODE
from .customized_rnn import RNN_TYPES, ConcatGRU
from .attention import ATTENTION_LAYER
from .hier_atten_rnn import HierAttenGRU
from .cnn import STACK_CNN_CLASS,Conv1d

RNN_TYPES = dict(RNN_TYPES)
RNN_TYPES['HierAttenGRU'] = HierAttenGRU

class FeatureBlock(BasicModel):
    def __init__(self,in_channels,num_layers,stack_cnn_class=None,**kwargs):
        super().__init__()
        self.in_channels=in_channels
        self.num_layers = num_layers
        stack_cnn_class = stack_cnn_class or 'ConcatCNN'
        self.stack_cnn_class = STACK_CNN_CLASS[stack_cnn_class]        
        self.cnns = self.stack_cnn_class(in_channels,self.num_layers,**kwargs)
        self.out_channels = self.cnns.out_channels
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = super().get_config()
        config['setting'] = self.cnns.get_config()
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths,_ = self.cnns(x,lengths)
        cnn_distribution = self.cnns.saved_distribution
        if self.save_distribution:
            self._distribution.update(cnn_distribution)
        return x,lengths

class RelationBlock(BasicModel):
    def __init__(self,in_channels,rnn_type,**kwargs):
        super().__init__()
        self.in_channels=in_channels
        if isinstance(rnn_type,str):
            try:
                self.rnn_type = RNN_TYPES[rnn_type]
            except:
                raise Exception("{} is not supported".format(rnn_type))
        else:        
            self.rnn_type = rnn_type
        self.rnn = self.rnn_type(in_channels=in_channels,**kwargs)
        self.out_channels = self.rnn.out_channels
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = super().get_config()
        config['setting'] = self.rnn.get_config()
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        if isinstance(x,list):
            for i in range(len(x)):
                self._distribution["pre_rnn_result_{}".format(i)] = x[i]
        else:
            self._distribution["pre_rnn_result"] = x
        rnn_distribution = {}
        x = self.rnn(x,lengths)
        rnn_distribution.update(self.rnn.saved_distribution)
        self._distribution["pre_last_result"] = x
        if self.save_distribution:
            self._distribution.update(rnn_distribution)
        return x,lengths

class ConnectBlock(BasicModel):
    def __init__(self,in_channels,compression_factor=None,dropout=None):
        super().__init__()
        self.in_channels = in_channels
        self.compression_factor = compression_factor or 1
        self.dropout = dropout or 0
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        self.out_channels = int(in_channels*self.compression_factor)
        self.transition = CANBlock(in_channels=in_channels,kernel_size=1,
                                   out_channels=self.out_channels,
                                   norm_mode=self.norm_mode,
                                   customized_init=customized_init)
        self.save_distribution = True
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config['compression_factor'] = self.compression_factor
        config['dropout'] = self.dropout
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        if self.compression_factor < 1:
            x,lengths,_ = self.transition(x,lengths)
        if self.dropout > 0:
            x = self.dropout_layer(x)
        return x,lengths
    
class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block):
        super().__init__()
        self.feature_block = feature_block
        self.relation_block = relation_block
        self.in_channels = self.feature_block.in_channels
        if relation_block is not None:
            self.out_channels = self.relation_block.out_channels
        else:
            self.out_channels = self.feature_block.out_channels
        self.save_distribution = True
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config['feature_block'] = self.feature_block.get_config()
        if self.relation_block is not None:
            config['relation_block'] = self.relation_block.get_config()
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        distribution = {}
        features,lengths = self.feature_block(x, lengths)
        distribution.update(self.feature_block.saved_distribution)
        x = features
        if self.relation_block is not None:
            x,lengths = self.relation_block(x, lengths)
            distribution.update(self.relation_block.saved_distribution)
        if self.save_distribution:
            self._distribution = distribution

        return x,lengths

class SeqAnnBuilder:
    def __init__(self):
        self.in_channels = None
        self.out_channels = None
        self.feature_block_config = None
        self.relation_block_config = None
        self.reset()

    @property
    def config(self):
        config = {}
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['feature_block_config'] = self.feature_block_config
        config['relation_block_config'] = self.relation_block_config
        return config
   
    @config.setter
    def config(self,config):
        self.feature_block_config = config['feature_block_config']
        self.relation_block_config = config['relation_block_config']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        
    def reset(self):
        self.feature_block_config = {'out_channels':16,'kernel_size':16,
                                     'num_layers':4,"norm_mode":"after_activation"}
        self.relation_block_config = {'num_layers':4,'hidden_size':16,
                                      'batch_first':True,'bidirectional':True,
                                      'rnn_type':'GRU'}
        self.in_channels = 4
        self.out_channels = 3
        
    def build(self):
        feature_block = FeatureBlock(self.in_channels,**self.feature_block_config)       
        relation_block_config = dict(self.relation_block_config)
        relation_block = None
        
        if relation_block_config['num_layers'] > 0 :
            relation_block = RelationBlock(feature_block.out_channels,
                                           **relation_block_config)
            out_channels = relation_block.out_channels
        else:    
            out_channels = feature_block.out_channels
        
        model = SeqAnnModel(feature_block,relation_block)
        return model
