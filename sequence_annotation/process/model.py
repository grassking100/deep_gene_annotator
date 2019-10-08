import torch
from torch import nn
from .customized_layer import Conv1d , BasicModel
from .rnn import GRU, LSTM,GRU_INIT_MODE
from .customized_rnn import RNN_TYPES, ConcatGRU
from .attention import ATTENTION_LAYER
from .hier_atten_rnn import HierAttenGRU
from .cnn import STACK_CNN_CLASS

RNN_TYPES = dict(RNN_TYPES)
RNN_TYPES['HierAttenGRU'] = HierAttenGRU

class FeatureBlock(BasicModel):
    def __init__(self,in_channels,num_layers,cnn_setting,
                 compression_factor=None,stack_cnn_class=None,
                 norm_mode=None,dropout=None,**kwargs):
        super().__init__()
        self.in_channels=in_channels
        self.cnn_setting = cnn_setting
        self.num_layers = num_layers
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
        if self.compression_factor < 1:
            in_num = self.cnns.out_channels
            out_num = int(in_num*self.compression_factor)
            self.transition = CANBlock(in_channels=in_num,kernel_size=1,
                                       out_channels=out_num,
                                       norm_mode=self.norm_mode)
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
        self._build_layers()
        self.reset_parameters()
        self.save_distribution = True

    def get_config(self):
        config = {}
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
                          kernel_size=self.kernel_size)
        self.customized_init = customized_init
        self.reset_parameters()

    def forward(self,x,lengths):
        x,lengths,_ = self.cnn(x,lengths)
        return x,lengths

    def get_config(self):
        config = {'out_channels':self.out_channels,
                  'kernel_size':self.kernel_size,
                  'in_channels':self.in_channels}
        return config
    
    def reset_parameters(self):
        super().reset_parameters()
        if self.customized_init is not None:
            self.customized_init(self.cnn,gain=1)
    
class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block,
                 project_layer,use_sigmoid=False,
                 predict_site_layer=None,predict_site_by=None,
                 attention_block=None):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.feature_block = feature_block
        self.relation_block = relation_block
        self.project_layer = project_layer
        self.attention_block = attention_block
        if project_layer is not None:
            out_channels = self.project_layer.out_channels
        elif relation_block is not None:
            out_channels = self.relation_block.out_channels
        else:
            out_channels = self.feature_block.out_channels
        self.out_channels = out_channels
        self.predict_site_layer = predict_site_layer
        if self.predict_site_layer is not None:
            self.out_channels += self.predict_site_layer.out_channels
        self.save_distribution = True
        self.predict_site = self.predict_site_layer is not None
        predict_site_by = predict_site_by or 'feature_block'
        if predict_site_by in ['feature_block','relation_block']:
            self._predict_site_by = predict_site_by
        else:
            raise Exception("Wrong block type, got {}".format(predict_site_by_block))
        self.reset_parameters()

    def get_config(self):
        config = {}
        config['use_sigmoid'] = self.use_sigmoid
        config['feature_block'] = self.feature_block.get_config()
        config['out_channels'] = self.out_channels
        config['in_channels'] = self.feature_block.in_channels
        config['predict_site_by'] = self._predict_site_by
        if self.attention_block is not None:
            config['attention_block'] = self.attention_block.get_config()
        if self.relation_block is not None:
            config['relation_block'] = self.relation_block.get_config()
        if self.project_layer is not None:
            config['project_layer'] = self.project_layer.get_config()
        if self.predict_site_layer is not None:
            config['predict_site_layer'] = self.predict_site_layer.get_config()
        return config

    def forward(self, x, lengths):
        #X shape : N,C,L
        distribution = {}
        features,lengths = self.feature_block(x, lengths)
        distribution.update(self.feature_block.saved_distribution)
        x = features
        if self.attention_block is not None:
            x = self.attention_block(x,lengths)
            distribution.update(self.attention_block.saved_distribution)
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
        
        if self.predict_site:
            if self._predict_site_by == 'feature_block':
                previous_signal = features
            else:
                previous_signal = relation
            site_predict,lengths = self.predict_site_layer(previous_signal,lengths)
            site_predict = nn.LogSoftmax(dim=1)(site_predict).exp()
            distribution["site_predict"] = site_predict
            x = torch.cat([ann,site_predict],1)
        else:
            x = ann
            
        if self.save_distribution:
            self._distribution = distribution

        return x,lengths

class Disrim(BasicModel):
    def __init__(self,label_channels,rnn_size,rnn_num=None,seq_channels=None,
                 cnn_out=None,kernel_size=None,**kwargs):
        super().__init__()
        self.seq_channels = seq_channels or 4
        self.label_channels = label_channels
        self.rnn_size = rnn_size or 16
        self.kernel_size = kernel_size or 3
        self.rnn_num = rnn_num or 1
        self.cnn_out = cnn_out or 1
        self.concat = Concat(dim=1)
        self.out_channels = 1
        if self.cnn_out > 0:
            self.cnn = CANBlock(self.label_channels,self.kernel_size,self.cnn_out,
                                norm_mode='after_activation',padding_handle='partial')
            in_channels = self.cnn.out_channels
        else:
            in_channels = self.label_channels
        
        in_channels += self.seq_channels
        self.rnn = GRU(in_channels=in_channels,hidden_size=self.rnn_size,bidirectional=True,
                       num_layers=self.rnn_num, batch_first=True,**kwargs)
        self.project = Conv1d(in_channels=self.rnn.out_channels,
                              out_channels=1,kernel_size=1)

    def get_config(self):
        config = {}
        config['seq_channels'] = self.seq_channels
        config['label_channels'] = self.label_channels
        config['kernel_size'] = self.kernel_size
        config['rnn_size'] = self.rnn_size
        config['rnn_num'] = self.rnn_num
        config['cnn_out'] = self.cnn_out
        config['out_channels'] = self.out_channels
        return config
        
    def forward(self, seq, label, lengths):
        seq = seq.float()
        label = label.float()
        if self.cnn_out > 0:
            label, lengths,_ = self.cnn(label,lengths)
        x,lengths = self.concat([seq,label],lengths)
        x = self.rnn(x,lengths)
        x,_,_ = self.project(x,lengths)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        return x
    
class SAGAN(BasicModel):
    """Sequence annotation GAN model"""
    def __init__(self,gan,discrim):
        super().__init__()
        self.gan = gan
        self.discrim = discrim
        
    @property
    def saved_distribution(self):
        return self.gan.saved_distribution
        
    def get_config(self):
        config = {'GAN':self.gan.get_config(),
                  'discriminator':self.discrim.get_config()}
        return config

class SeqAnnBuilder:
    def __init__(self):
        self.feature_block_config = None
        self.relation_block_config = None
        self.project_layer_config = None
        self.in_channels = None
        self.out_channels = None
        self.use_sigmoid = None
        self.discrim_config = None
        self.use_discrim = None
        self.site_ann_method = None
        self.predict_site_by = None
        self.use_attention = None
        self.attention_config = None
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
        config['discrim_config'] = self.discrim_config
        config['use_discrim'] = self.use_discrim
        config['site_ann_method'] = self.site_ann_method
        config['predict_site_by'] = self.predict_site_by
        config['use_attention'] = self.use_attention
        config['attention_config'] = self.attention_config
        return config
   
    @config.setter
    def config(self,config):
        self.feature_block_config = config['feature_block_config']
        self.relation_block_config = config['relation_block_config']
        self.project_layer_config = config['project_layer_config']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.use_sigmoid = config['use_sigmoid']
        self.discrim_config = config['discrim_config']
        self.use_discrim = config['use_discrim']
        self.site_ann_method = config['site_ann_method']
        self.predict_site_by = config['predict_site_by']
        self.use_attention = config['use_attention']
        self.attention_config = config['attention_config']
        
    def reset(self):
        self.feature_block_config = {'cnn_setting':{'out_channels':16,'kernel_size':16},
                                     'num_layers':4,"norm_mode":"after_activation"}
        self.relation_block_config = {'rnn_setting':{'num_layers':4,'hidden_size':16,
                                                     'batch_first':True,
                                                     'bidirectional':True},
                                      'rnn_type':'GRU'}
        self.discrim_config = {'kernel_size':32}
        self.attention_config = {'attention_type':'BiRNNAttention','attention_setting':{}}
        self.project_layer_config = {}
        self.in_channels = 4
        self.out_channels = 3
        self.use_sigmoid = True
        self.use_discrim = False
        self.site_ann_method = None
        self.use_attention = False
        
    def build(self):
        feature_block = FeatureBlock(self.in_channels,**self.feature_block_config)
        attention_block = None        
        relation_block_config = dict(self.relation_block_config)
        project_layer_config = dict(self.project_layer_config)
        relation_init_func = None
        project_layer_init_func = None
        relation_block = None
        project_layer = None
        predict_site_layer = None

        if 'customized_gru_init_mode' in relation_block_config['rnn_setting']:
            mode = relation_block_config['rnn_setting']['customized_gru_init_mode']
            relation_init_func = GRU_INIT_MODE[mode]
            
        relation_block_config['rnn_setting']['customized_init'] = relation_init_func

        if 'customized_gru_init_mode' in relation_block_config['rnn_setting']:
            del relation_block_config['rnn_setting']['customized_gru_init_mode']

        if 'customized_init' in project_layer_config and project_layer_config['customized_init']:
            project_layer_init_func = customized_init_cnn
            
        project_layer_config['customized_init'] = project_layer_init_func

        if self.use_attention:
            attention_class = ATTENTION_LAYER[self.attention_config['attention_type']]
            attention_block = attention_class(feature_block.out_channels,**self.attention_config['attention_setting'])
        
        if relation_block_config['rnn_setting']['num_layers'] > 0 :
            relation_block = RelationBlock(feature_block.out_channels,
                                           **relation_block_config)
            out_channels = relation_block.out_channels
        else:    
            out_channels = feature_block.out_channels
        
        if self.project_layer_config['kernel_size'] >=1:
            project_layer = ProjectLayer(out_channels,self.out_channels,**project_layer_config)
            
        if self.site_ann_method is not None:
            if self.predict_site_by == 'feature_block':
                in_channels = feature_block.out_channels
            else:
                in_channels = relation_block.out_channels
            if self.site_ann_method == 'positive':
                out_channels = 4
            else:
                out_channels = 5
            predict_site_layer = ProjectLayer(in_channels,out_channels,**project_layer_config)
        
        model = SeqAnnModel(feature_block,relation_block,project_layer,
                            use_sigmoid=self.use_sigmoid,
                            predict_site_layer=predict_site_layer,
                            predict_site_by=self.predict_site_by,
                            attention_block=attention_block)
        if self.use_discrim:
            gan = model
            discrim = Disrim(model.out_channels,**self.discrim_config).cuda()
            model = SAGAN(model,discrim)
        return model
