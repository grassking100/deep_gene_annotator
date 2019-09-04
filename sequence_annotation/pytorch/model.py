import torch
from torch import nn
from torch.nn.init import normal_,constant_
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn import ReLU
from .noisy_activation import NoisyReLU
from .customized_layer import Conv1d, Concat,PaddedBatchNorm1d, BasicModel, add
from .rnn import GRU,LSTM
from .customized_rnn import RNN_TYPES, ConcatGRU

ACTIVATION_FUNC = {'NoisyReLU':NoisyReLU(),'ReLU':ReLU()}

class CANBlock(BasicModel):
    def __init__(self,in_channels,kernel_size,out_channels,
                 norm_mode=None,norm_type=None,
                 activation_function=None):
        super().__init__()
        self.name = ""
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.norm_type = norm_type
        if norm_mode in ["before_cnn","after_cnn","after_activation",None]:
            self.norm_mode = norm_mode
        else:
            raise Exception('norm_mode should be "before_cnn", "after_cnn", None, or "after_activation"')
        if activation_function is None:
            self.activation_function = NoisyReLU()
        else:
            if isinstance(str,activation_function):
                self.activation_function = ACTIVATION_FUNC[activation_function]
            else:
                self.activation_function = activation_function
        use_bias = self.norm_mode != "after_cnn"
        self.cnn = Conv1d(in_channels=self.in_channels,kernel_size=self.kernel_size,
                          out_channels=self.out_channels,bias=use_bias)
        if self.norm_mode is not None:
            in_channel = {"before_cnn":self.in_channels,"after_cnn":self.out_channels,
                          "after_activation":self.out_channels}
            if self.norm_type is None:
                self.norm = PaddedBatchNorm1d(in_channel[self.norm_mode])
            else:
                self.norm = self.norm_type(in_channel[self.norm_mode])
        
        self._distribution = {}
        self.reset_parameters()

    def _normalized(self,x,lengths):
        x = self.norm(x,lengths)
        self._distribution['norm_{}'.format(self.name)] = x
        return x

    def forward(self, x, lengths):
        #X shape : N,C,L
        if self.norm_mode=='before_cnn':
            x = self._normalized(x,lengths)
        x,lengths = self.cnn(x,lengths)
        self._distribution['cnn_x_{}'.format(self.name)] = x
        if self.norm_mode=='after_cnn':
            x = self._normalized(x,lengths)
        x = self.activation_function(x)
        self._distribution['post_act_x_{}'.format(self.name)] = x
        if self.norm_mode=='after_activation':
            x = self._normalized(x,lengths)
        return x,lengths
    
    def get_config(self):
        config = {}
        config['name'] = self.name
        config['in_channels'] = self.in_channels
        config['kernel_size'] = self.kernel_size
        config['out_channels'] = self.out_channels
        config['norm_type'] = self.norm_type
        config['norm_mode'] = self.norm_mode
        config['activation_function'] = str(self.activation_function)
        return config

class StackCNN(BasicModel):
    def __init__(self,in_channels,num_layers,cnn_setting):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.cnn_setting = cnn_setting
    
    def get_config(self):
        config = {'cnn_setting':self.cnn_setting,
                  'num_layers':self.num_layers,
                  'in_channels':self._in_channels,
                  'out_channels':self.out_channels}
        return config
    
class ConcatCNN(StackCNN):
    def __init__(self,in_channels,num_layers,cnn_setting):
        super().__init__(in_channels,num_layers,cnn_setting)
        in_num = in_channels
        self.cnns = []
        for index in range(self.num_layers):
            setting = {}
            for key,value in cnn_setting.items():
                if isinstance(value,list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,**setting)
            cnn.name=str(index)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            in_num += cnn.out_channels
        self.out_channels = in_num
        self.concat = Concat(dim=1)
        self.reset_parameters()

    def forward(self, x, lengths):
        #X shape : N,C,L
        for index in range(self.num_layers):
            pre_x = x
            cnn = self.cnns[index]
            x,lengths = cnn(x,lengths)
            self._distribution.update(cnn.saved_distribution)
            x,lengths = self.concat([pre_x,x],lengths)
        self._distribution['cnn_result'] = x
        return x,lengths
    
class ResCNN(StackCNN):
    def __init__(self,in_channels,num_layers,cnn_setting):
        super().__init__(in_channels,num_layers,cnn_setting)
        in_num = in_channels
        self.cnns = []
        for index in range(self.num_layers):
            setting = {}
            for key,value in cnn_setting.items():
                if isinstance(value,list):
                    setting[key] = value[index]
                else:
                    setting[key] = value
            cnn = CANBlock(in_num,**setting)
            cnn.name=str(index)
            self.cnns.append(cnn)
            setattr(self, 'cnn_'+str(index), cnn)
            in_num = cnn.out_channels
        self.out_channels = in_num
        self.reset_parameters()

    def forward(self, x, lengths):
        #X shape : N,C,L
        for index in range(self.num_layers):
            pre_x = x
            cnn = self.cnns[index]
            x,lengths = cnn(x,lengths)
            self._distribution.update(cnn.saved_distribution)
            if index > 0:
                x,lengths = add(pre_x,x,lengths)
        self._distribution['cnn_result'] = x
        return x,lengths

class ProjectLayer(BasicModel):
    def __init__(self,in_channels,out_channels,kernel_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size or 1
        self.cnn = Conv1d(in_channels=in_channels,out_channels=out_channels,
                          kernel_size=self.kernel_size)

    def forward(self,x,lengths):
        x = self.cnn(x)
        return x,lengths

    def get_config(self):
        config = {'out_channels':self.out_channels,
                  'kernel_size':self.kernel_size,
                  'in_channels':self.in_channels}
        return config

STACK_CNN_CLASS = {'ConcatCNN':ConcatCNN,'ResCNN':ResCNN}
    
class FeatureBlock(BasicModel):
    def __init__(self,in_channels,num_layers,cnn_setting,reduce_cnn_ratio=None,
                 reduced_cnn_number=None,stack_cnn_class=None):
        super().__init__()
        self.in_channels=in_channels
        self.cnn_setting = cnn_setting
        self.num_layers = num_layers
        stack_cnn_class = stack_cnn_class or 'ConcatCNN'
        self.stack_cnn_class = STACK_CNN_CLASS[stack_cnn_class]
            
        if self.num_layers <= 0:
            raise Exception("CNN layer size sould be positive")
        self.reduced_cnn_number = reduced_cnn_number
        self.reduce_cnn_ratio = reduce_cnn_ratio or 1
        if self.reduce_cnn_ratio <= 0:
            raise Exception("Reduce_cnn_ratio should be larger than zero")
        self._reduce_cnn = self.reduce_cnn_ratio < 1 or self.reduced_cnn_number is not None
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
        config['reduce_cnn_ratio'] = self.reduce_cnn_ratio
        config['num_layers'] = self.num_layers
        config['reduced_cnn_number'] = self.reduced_cnn_number
        config['out_channels'] = self.out_channels
        config['stack_cnn_class'] = self.stack_cnn_class.__name__
        return config

    def _build_layers(self):
        in_channels=self.in_channels
        self.cnns = self.stack_cnn_class(in_channels,self.num_layers,self.cnn_setting)
        if self._reduce_cnn:
            if self.reduce_cnn_ratio < 1:
                hidden_size = int(self.cnns.hidden_size*self.reduce_cnn_ratio)
            else:
                hidden_size = self.reduced_cnn_number
            self.cnn_merge = Conv1d(in_channels=self.cnns.hidden_size,
                                    kernel_size=1,out_channels=hidden_size)
            in_channels = hidden_size
        else:
            in_channels = self.cnns.out_channels
        self.out_channels = in_channels

    def forward(self, x, lengths):
        #X shape : N,C,L
        x,lengths = self.cnns(x,lengths)
        cnn_distribution = self.cnns.saved_distribution
        if self._reduce_cnn:
            x,lengths = self.cnn_merge(x,lengths)
        if self.save_distribution:
            self._distribution.update(cnn_distribution)
        return x,lengths

class RelationBlock(BasicModel):
    def __init__(self,in_channels,rnn_setting,rnn_type=None):
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
        rnn_setting = self.rnn_setting
        if 'rnn_type' in rnn_setting.keys():
            rnn_setting['rnn_type'] = rnn_setting['rnn_type'].__name__
        config['rnn_setting'] = rnn_setting
        config['rnn_type'] = self.rnn_type.__name__
        return config

    def _build_layers(self):
        in_channels=self.in_channels
        self.rnn = self.rnn_type(in_channels=in_channels,**self.rnn_setting)
        self.out_channels = self.rnn.out_channels

    def forward(self, x, lengths):
        #X shape : N,C,L
        self._distribution["pre_rnn_result"] = x
        rnn_distribution = {}
        x = self.rnn(x,lengths)
        self._distribution["pre_last_result"] = x
        if self.save_distribution:
            self._distribution.update(rnn_distribution)
        return x,lengths

class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block,
                 project_layer,use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.feature_block = feature_block
        self.relation_block = relation_block
        self.project_layer = project_layer
        self.out_channels = self.project_layer.out_channels
        self.save_distribution = True
        self.reset_parameters()

    def get_config(self):
        config = {}
        config['use_sigmoid'] = self.use_sigmoid
        config['feature_block'] = self.feature_block.get_config()
        config['relation_block'] = self.relation_block.get_config()
        config['project_layer'] = self.project_layer.get_config()
        config['out_channels'] = self.out_channels
        config['in_channels'] = self.feature_block.in_channels
        return config

    def forward(self, x, lengths,return_length=False):
        #X shape : N,C,L
        x,lengths = self.feature_block(x, lengths)
        x,lengths = self.relation_block(x, lengths)
        distribution = {}
        distribution["pre_last"] = x
        x,lengths = self.project_layer(x,lengths)
        distribution.update(self.feature_block.saved_distribution)
        distribution.update(self.relation_block.saved_distribution)
        distribution["last"] = x
        if not self.use_sigmoid:
            x = nn.LogSoftmax(dim=1)(x).exp()
            distribution["softmax"] = x
        else:
            x = torch.sigmoid(x)
            distribution["sigmoid"] = x
            
        if self.save_distribution:
            self._distribution = distribution

        if return_length:
            return x,lengths
        else:
            return x

class Disrim(BasicModel):
    def __init__(self,label_channels,rnn_size,rnn_num=None,seq_channels=None,
                 cnn_out=None,kernel_size=None,**kwargs):
        super().__init__()
        self.seq_channels = seq_channels or 4
        self.label_channels = label_channels
        self.rnn_size = rnn_size
        self.kernel_size = kernel_size or 3
        self.rnn_num = rnn_num or 1
        self.cnn_out = cnn_out or 1
        self.concat = Concat(dim=1)
        self.out_channels = 1
        if self.cnn_out > 0:
            self.cnn = CANBlock(self.label_channels,self.kernel_size,self.cnn_out,
                                norm_mode='after_activation')
            in_channels = self.cnn.out_channels
        else:
            in_channels = self.label_channels
        
        in_channels += self.seq_channels
        self.rnn = ConcatGRU(in_channels,self.rnn_size,bidirectional=True,
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
            label, lengths = self.cnn(label,lengths)
        x,lengths = self.concat([seq,label],lengths)
        x = self.rnn(x,lengths)
        x = self.project(x)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        return x
    
class SAGAN(BasicModel):
    """Sequence annotation GAN model"""
    def __init__(self,gan,discrim):
        super().__init__()
        self.gan = gan
        self.discrim = discrim
        self._distribution = self.gan.saved_distribution
        
    def get_config(self):
        config = {'GAN':self.gan.get_config(),
                  'discriminator':self.discrim.get_config()}
        return config
        
def seq_ann_inference(outputs,mask):
    """
        Data shape is N,C,L
        Input channel order: Transcription potential, Intron potential
        Output channel order: Exon, Intron , Other
    """
    if outputs.shape[1]!=2:
        raise Exception("Channel size should be two")
    transcript_potential = outputs[:,0,:].unsqueeze(1)
    intron_potential = outputs[:,1,:].unsqueeze(1)
    other = 1-transcript_potential
    if mask is not None:
        mask = mask[:,:outputs.shape[2]].unsqueeze(1)
        other = other * mask.float()
    transcript_mask = (transcript_potential>=0.5).float()
    intron = transcript_mask * intron_potential
    exon = transcript_mask * (1-intron_potential)
    result = torch.cat([exon,intron,other],dim=1)
    return result

def seq_ann_reverse_inference(outputs,mask):
    """
        Data shape is N,C,L
        Output channel order: Exon, Intron , Other
        Output channel order: Transcription potential, Intron potential
    """
    if outputs.shape[1] != 3:
        raise Exception("Channel size should be two")
    intron_potential = outputs[:,1,:].unsqueeze(1)
    other_potential = outputs[:,2,:].unsqueeze(1)
    transcript_potential = 1 - other_potential
    result = torch.cat([transcript_potential,intron_potential],dim=1)
    return result

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
        
    def reset(self):
        self.feature_block_config = {'cnn_setting':{"norm_mode":"after_activation",
                                                     'out_channels':16,
                                                     'kernel_size':16},
                                     'num_layers':4}
        self.relation_block_config = {'rnn_setting':{'num_layers':4,'hidden_size':16,
                                                     'batch_first':True,
                                                     'bidirectional':True},
                                      'rnn_type':'GRU'}
        self.discrim_config = {'rnn_num':16}
        self.project_layer_config = {}
        self.in_channels = 4
        self.out_channels = 3
        self.use_sigmoid = True
        self.use_discrim = False
        
    def build(self):
        feature_block = FeatureBlock(self.in_channels,**self.feature_block_config)
        relation_block = RelationBlock(feature_block.out_channels,
                                       **self.relation_block_config)
        project_layer = ProjectLayer(relation_block.out_channels,self.out_channels,
                                     **self.project_layer_config)
        
        model = SeqAnnModel(feature_block,relation_block,project_layer,
                            use_sigmoid=self.use_sigmoid)
        if self.use_discrim:
            gan = model
            discrim = Disrim(self.out_channels,**self.discrim_config).cuda()
            model = SAGAN(model,discrim)
        return model
