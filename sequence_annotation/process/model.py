from .customized_layer import BasicModel
from .customized_rnn import RNN_TYPES
from .filter import FilteredRNN
from .hier_rnn import HierRNNBuilder
from .cnn import STACK_CNN_CLASS

class SeqAnnModel(BasicModel):
    def __init__(self,feature_block,relation_block):
        super().__init__()
        self.feature_block = feature_block    
        self.relation_block = relation_block
        self.in_channels = self.relation_block.in_channels
        
        if self.feature_block is not None:
            self.in_channels = self.feature_block.in_channels
            
        self.out_channels = self.relation_block.out_channels
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        if self.feature_block is not None:
            config['feature_block'] = self.feature_block.get_config()
        config['relation_block'] = self.relation_block.get_config()
        return config

    def forward(self, features, lengths,answers=None):
        #shape : N,C,L
        if self.feature_block is not None:
            features,lengths,_ = self.feature_block(features, lengths)
            self.update_distribution(self.feature_block.saved_distribution)
        result = self.relation_block(features, lengths,answers=answers)
        self.update_distribution(self.relation_block.saved_distribution)
        return result,lengths

class SeqAnnBuilder:
    def __init__(self,in_channels=None):
        self._in_channels = in_channels or 4
        self._out_channels = 3
        self._output_act = 'softmax'
        self._rnn_type = 'ProjectedRNN'
        self._filter_num = None
        self._filter_hidden = None
        self._use_first_filter = False
        self._use_second_filter = False
        self._use_common_filter = False
        self._use_feature_block = True
        self._stack_cnn_class = 'ConcatCNN'
        self._feature_block_config = {'out_channels':16,'kernel_size':16,
                                      'num_layers':4,"norm_mode":"after_activation"}

        self._relation_block_config = {'num_layers':4,'hidden_size':16,
                                       'batch_first':True,'bidirectional':True}
        
    def update_feature_block(self,stack_cnn_class=None,**config):
        self._feature_block_config.update(config)
        self._use_feature_block = self._feature_block_config['num_layers'] != 0
        self._stack_cnn_class = stack_cnn_class or self._stack_cnn_class

    def update_relation_block(self,out_channels=None,rnn_type=None,
                              filter_num=None,filter_hidden=None,
                              use_first_filter=None,use_second_filter=None,
                              use_common_filter=None,
                              output_act=None,**config):
        self._output_act = output_act or self._output_act
        self._rnn_type = rnn_type or self._rnn_type
        self._out_channels = out_channels or self._out_channels
        self._filter_num = filter_num or self._filter_num
        self._filter_hidden = filter_hidden or self._filter_hidden

        if use_first_filter is not None:
            self._use_first_filter = use_first_filter
            
        if use_second_filter is not None:
            self._use_second_filter = use_second_filter
            
        if use_common_filter is not None:
            self._use_common_filter = use_common_filter
        self._relation_block_config.update(config)
        
    def build(self):
        in_channels = self._in_channels
        feature_block = None
        if self._use_feature_block:
            feature_block = STACK_CNN_CLASS[self._stack_cnn_class](self._in_channels,**self._feature_block_config)
            in_channels = feature_block.out_channels

        if self._rnn_type in RNN_TYPES:
            rnn_class = RNN_TYPES[self._rnn_type]
            relation_block = rnn_class(in_channels,output_act=self._output_act,
                                       out_channels=self._out_channels,**self._relation_block_config)

        elif self._rnn_type == 'FilteredRNN':
            relation_block = FilteredRNN(in_channels,output_act=self._output_act,
                                         out_channels=self._out_channels,**self._relation_block_config)

        elif self._rnn_type == 'HierRNN':
            builder = HierRNNBuilder(in_channels,output_act='sigmoid',**self._relation_block_config)
            builder.set_filter_place(first=self._use_first_filter,second=self._use_second_filter,
                                     common=self._use_common_filter)

            builder.set_filter_setting(hidden_size=self._filter_hidden,num_layers=self._filter_num)
            relation_block = builder.build()
        else:
            raise Exception("{} is not supported".format(self._rnn_type))

        model = SeqAnnModel(feature_block,relation_block)
        return model
