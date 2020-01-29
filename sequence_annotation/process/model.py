import torch
from .customized_layer import BasicModel
from .customized_rnn import RNN_TYPES
from .hier_atten_rnn import HierAttenGRU,AttenGRU,HierGRU
from .cnn import STACK_CNN_CLASS

RNN_TYPES = dict(RNN_TYPES)
RNN_TYPES['HierAttenGRU'] = HierAttenGRU
RNN_TYPES['HierGRU'] = HierGRU
RNN_TYPES['AttenGRU'] = AttenGRU

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

    def get_config(self):
        config = super().get_config()
        config['setting'] = self.cnns.get_config()
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths,_ = self.cnns(x,lengths)
        self.update_distribution(self.cnns.saved_distribution)
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

    def get_config(self):
        config = super().get_config()
        config['setting'] = self.rnn.get_config()
        return config

    def forward(self, x, lengths,answers=None):
        #X shape : N,C,L
        if isinstance(x,list):
            for i in range(len(x)):
                self.update_distribution(x[i],key="pre_rnn_result_{}".format(i))
        else:
            self.update_distribution(x,key="pre_rnn_result")
        x = self.rnn(x,lengths,answers=answers)
        self.update_distribution(self.rnn.saved_distribution)
        self.update_distribution(x,key="pre_last_result")
        return x,lengths
    
class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block=None,last_act=None):
        super().__init__()
        if last_act == 'none':
            self.last_act=None
        else:
            self.last_act=last_act
        self.feature_block = feature_block
        self.relation_block = relation_block
        self.in_channels = self.feature_block.in_channels
        if relation_block is not None:
            self.out_channels = self.relation_block.out_channels
        else:
            self.out_channels = self.feature_block.out_channels
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config['last_act'] = self.last_act
        config['feature_block'] = self.feature_block.get_config()
        if self.relation_block is not None:
            config['relation_block'] = self.relation_block.get_config()
        return config

    def forward(self, x, lengths,answers=None):
        #X shape : N,C,L
        features,lengths = self.feature_block(x, lengths)
        self.update_distribution(self.feature_block.saved_distribution)
        x = features
        if self.relation_block is not None:
            x,lengths = self.relation_block(x, lengths,answers=answers)
            self.update_distribution(self.relation_block.saved_distribution)
        if self.last_act is not None:
            if self.last_act == 'softmax':
                x = torch.softmax(x,1)
            elif self.last_act == 'sigmoid':
                x = torch.sigmoid(x)
            else:
                raise Exception("Wrong activation name, {}".format(self.last_act))
        return x,lengths

class SeqAnnBuilder:
    def __init__(self):
        self.in_channels = None
        self.out_channels = None
        self.last_act = None
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
        config['last_act'] = self.last_act
        return config
   
    @config.setter
    def config(self,config):
        self.feature_block_config = config['feature_block_config']
        self.relation_block_config = config['relation_block_config']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.last_act = config['last_act']
        
    def reset(self):
        self.feature_block_config = {'out_channels':16,'kernel_size':16,
                                     'num_layers':4,"norm_mode":"after_activation"}
        self.relation_block_config = {'num_layers':4,'hidden_size':16,
                                      'batch_first':True,'bidirectional':True,
                                      'rnn_type':'ProjectedGRU'}
        self.in_channels = 4
        self.out_channels = 3
        self.last_act = None
        
    def build(self):
        feature_block = FeatureBlock(self.in_channels,**self.feature_block_config)
        relation_block = None
        
        if self.relation_block_config['num_layers'] > 0 :
            relation_block = RelationBlock(feature_block.out_channels,
                                           out_channels=self.out_channels,
                                           **self.relation_block_config)
        
        model = SeqAnnModel(feature_block,relation_block,last_act=self.last_act)
        return model
