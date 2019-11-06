import torch
from torch import nn
from .customized_layer import BasicModel
from .rnn import GRU, LSTM,GRU_INIT_MODE
from .customized_rnn import RNN_TYPES, ConcatGRU
from .attention import ATTENTION_LAYER
from .hier_atten_rnn import HierAttenGRU,GatedStackGRU
from .cnn import STACK_CNN_CLASS,Conv1d

RNN_TYPES = dict(RNN_TYPES)
RNN_TYPES['GatedStackGRU'] = GatedStackGRU
RNN_TYPES['HierAttenGRU'] = HierAttenGRU

class FeatureBlock(BasicModel):
    def __init__(self,in_channels,num_layers,cnn_setting,
                 compression_factor=None,stack_cnn_class=None,
                 norm_mode=None,dropout=None,**kwargs):
        super().__init__()
        self.in_channels=in_channels
        self.cnn_setting = cnn_setting
        self.num_layers = num_layers
        self.cnns = None
        stack_cnn_class = stack_cnn_class or 'ConcatCNN'
        self.norm_mode = norm_mode or 'after_activation'
        self.stack_cnn_class = STACK_CNN_CLASS[stack_cnn_class]
        self.kwargs = kwargs
        if self.num_layers <= 0:
            raise Exception("CNN layer size sould be positive")
        self.compression_factor = compression_factor or 1
        if self.compression_factor <= 0:
            raise Exception("The compression_factor should be larger than zero")
        self.dropout = dropout or 0
        self._build_layers()
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = {}
        config['in_channels'] = self.in_channels
        setting = self.cnn_setting
        for type_ in ['norm_type','activation_function']:
            if type_ in setting.keys():
                setting[type_] = str(setting[type_])
        config['cnn_setting'] = setting
        config['compression_factor'] = self.compression_factor
        config['num_layers'] = self.num_layers
        config['out_channels'] = self.out_channels
        config['stack_cnn_class'] = self.stack_cnn_class.__name__
        config['dropout'] = self.dropout
        return config

    def _build_layers(self):
        in_channels=self.in_channels
        self.cnns = self.stack_cnn_class(in_channels,self.num_layers,self.cnn_setting,
                                         norm_mode=self.norm_mode,**self.kwargs)
        customized_init = None
        if 'customized_init' in self.cnn_setting:
            customized_init = self.cnn_setting['customized_init']
        if self.compression_factor < 1:
            in_num = self.cnns.out_channels
            out_num = int(in_num*self.compression_factor)
            self.transition = CANBlock(in_channels=in_num,kernel_size=1,
                                       out_channels=out_num,
                                       norm_mode=self.norm_mode,
                                       customized_init=customized_init)
            in_channels = self.transition.out_channels
        else:
            in_channels = self.cnns.out_channels
        self.out_channels = in_channels
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths,_ = self.cnns(x,lengths)
        cnn_distribution = self.cnns.saved_distribution
        if self.compression_factor < 1:
            x,lengths,_ = self.transition(x,lengths)
        if self.dropout > 0:
            x = self.dropout_layer(x)
        if self.save_distribution:
            self._distribution.update(cnn_distribution)
        return x,lengths

class RelationBlock(BasicModel):
    def __init__(self,in_channels,rnn_setting,rnn_type):
        super().__init__()
        self.in_channels=in_channels
        self.rnn_setting = rnn_setting
        self.rnn_num = self.rnn_setting['num_layers']
        if self.rnn_setting['num_layers'] <= 0:
            raise Exception("RNN layer size sould be positive")
        if isinstance(rnn_type,str):
            try:
                self.rnn_type = RNN_TYPES[rnn_type]
            except:
                raise Exception("{} is not supported".format(rnn_type))
        else:        
            self.rnn_type = rnn_type
        self.rnn = None
        self._build_layers()
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = dict(self.rnn.get_config())
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        rnn_setting = dict(self.rnn_setting)
        if 'rnn_type' in rnn_setting.keys():
            rnn_setting['rnn_type'] = rnn_setting['rnn_type'].__name__
        if 'customized_init' in rnn_setting.keys():
            rnn_setting['customized_init'] = str(rnn_setting['customized_init'])
        config['rnn_setting'] = rnn_setting
        config['rnn_type'] = self.rnn_type.__name__
        return config

    def _build_layers(self):
        in_channels=self.in_channels
        self.rnn = self.rnn_type(in_channels=in_channels,**self.rnn_setting)
        self.out_channels = self.rnn.out_channels

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

class ProjectLayer(BasicModel):
    def __init__(self,in_channels,out_channels,kernel_size=None,customized_init=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if kernel_size is None:
            self.kernel_size = 1
        else:
            self.kernel_size = kernel_size
        self.cnn = Conv1d(in_channels=in_channels,out_channels=out_channels,
                          kernel_size=self.kernel_size,customized_init=customized_init)
        self.reset_parameters()

    def forward(self,x,lengths):
        x,lengths,_ = self.cnn(x,lengths)
        return x,lengths

    def get_config(self):
        config = {'out_channels':self.out_channels,
                  'kernel_size':self.kernel_size,
                  'in_channels':self.in_channels}
        return config
    
class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block,
                 project_layer,use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.feature_block = feature_block
        self.relation_block = relation_block
        self.project_layer = project_layer
        if project_layer is not None:
            out_channels = self.project_layer.out_channels
        elif relation_block is not None:
            out_channels = self.relation_block.out_channels
        else:
            out_channels = self.feature_block.out_channels
        self.out_channels = out_channels
        self.save_distribution = True
        self.reset_parameters()

    def get_config(self):
        config = {}
        config['use_sigmoid'] = self.use_sigmoid
        config['feature_block'] = self.feature_block.get_config()
        config['out_channels'] = self.out_channels
        config['in_channels'] = self.feature_block.in_channels
        if self.relation_block is not None:
            config['relation_block'] = self.relation_block.get_config()
        if self.project_layer is not None:
            config['project_layer'] = self.project_layer.get_config()
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        distribution = {}
        features,lengths = self.feature_block(x, lengths)
        distribution.update(self.feature_block.saved_distribution)
        x = features
        relation = None
        if self.relation_block is not None:
            relation,lengths = self.relation_block(x, lengths)
            distribution.update(self.relation_block.saved_distribution)
            x = relation
        
        distribution["pre_last"] = x
        if self.project_layer is not None:
            x,lengths = self.project_layer(x,lengths)
        distribution["last"] = x
        
        if self.project_layer is not None:
            if self.use_sigmoid:
                ann = torch.sigmoid(x)
                distribution["sigmoid"] = ann
            else:
                ann = nn.LogSoftmax(dim=1)(x).exp()
                distribution["softmax"] = ann
        else:
            ann = x
        x = ann 
        if self.save_distribution:
            self._distribution = distribution

        return x,lengths

class SeqAnnBuilder:
    def __init__(self):
        self.feature_block_config = None
        self.relation_block_config = None
        self.project_layer_config = None
        self.in_channels = None
        self.out_channels = None
        self.use_sigmoid = None
        self.reset()

    @property
    def config(self):
        config = {}
        config['feature_block_config'] = self.feature_block_config
        config['relation_block_config'] = self.relation_block_config
        config['project_layer_config'] = self.project_layer_config
        config['in_channels'] = self.in_channels
        config['out_channels'] = self.out_channels
        config['use_sigmoid'] = self.use_sigmoid
        return config
   
    @config.setter
    def config(self,config):
        self.feature_block_config = config['feature_block_config']
        self.relation_block_config = config['relation_block_config']
        self.project_layer_config = config['project_layer_config']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.use_sigmoid = config['use_sigmoid']
        
    def reset(self):
        self.feature_block_config = {'cnn_setting':{'out_channels':16,'kernel_size':16},
                                     'num_layers':4,"norm_mode":"after_activation"}
        self.relation_block_config = {'rnn_setting':{'num_layers':4,'hidden_size':16,
                                                     'batch_first':True,
                                                     'bidirectional':True},
                                      'rnn_type':'GRU'}
        self.project_layer_config = {'kernel_size':1}
        self.in_channels = 4
        self.out_channels = 3
        self.use_sigmoid = True
        
    def build(self):
        feature_block = FeatureBlock(self.in_channels,**self.feature_block_config)       
        relation_block_config = dict(self.relation_block_config)
        project_layer_config = dict(self.project_layer_config)
        relation_block = None
        project_layer = None
        
        if relation_block_config['rnn_setting']['num_layers'] > 0 :
            relation_block = RelationBlock(feature_block.out_channels,
                                           **relation_block_config)
            out_channels = relation_block.out_channels
        else:    
            out_channels = feature_block.out_channels
        
        if self.project_layer_config['kernel_size'] >=1:
            project_layer = ProjectLayer(out_channels,self.out_channels,**project_layer_config)

        model = SeqAnnModel(feature_block,relation_block,project_layer,
                            use_sigmoid=self.use_sigmoid)
        return model
