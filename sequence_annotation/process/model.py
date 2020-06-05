import torch
from ..utils.utils import read_json
from .customized_layer import BasicModel, generate_norm_class
from .customized_rnn import RNN_TYPES
from .filter import FilteredRNN
from .hier_rnn import HierRNNBuilder
from .cnn import STACK_CNN_CLASS


class SeqAnnModel(BasicModel):
    def __init__(self, feature_block, relation_block, norm_input_block=None):
        super().__init__()
        self.norm_input_block = norm_input_block
        self.feature_block = feature_block

        if self.feature_block is not None:
            self.in_channels = self.feature_block.in_channels
        else:
            self.in_channels = self.relation_block.in_channels

        self.relation_block = relation_block
        self.out_channels = self.relation_block.out_channels
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        if self.norm_input_block is not None:
            config['norm_input_block'] = self.norm_input_block.get_config()
        if self.feature_block is not None:
            config['feature_block'] = self.feature_block.get_config()
        config['relation_block'] = self.relation_block.get_config()
        return config

    def forward(self, features, lengths, answers=None):
        # shape : N,C,L
        if self.norm_input_block is not None:
            features = self.norm_input_block(features, lengths)

        if self.feature_block is not None:
            features, lengths, _ = self.feature_block(features, lengths)
            self.update_distribution(self.feature_block.saved_distribution)

        result = self.relation_block(features, lengths, answers=answers)
        self.update_distribution(self.relation_block.saved_distribution)
        self.update_distribution(result, key='last')
        return result, lengths


class SeqAnnBuilder:
    def __init__(self):
        self._in_channels = 4
        self._out_channels = 3
        self._output_act = 'softmax'
        self._rnn_type = 'ProjectedRNN'
        self._filter_num = None
        self._filter_hidden = None
        self._use_first_filter = False
        self._use_second_filter = False
        self._use_common_filter = False
        self._use_feature_block = True
        self.use_input_norm = True
        self._stack_cnn_class = 'ConcatCNN'
        self._feature_block_config = {
            'out_channels': 16,
            'kernel_size': 16,
            'num_layers': 4,
            "norm_mode": "after_activation"
        }

        self._relation_block_config = {
            'num_layers': 4,
            'hidden_size': 16,
            'batch_first': True,
            'bidirectional': True
        }

        self._norm_config = {
            'norm_class': 'PaddedBatchNorm1d',
            'momentum': 0.1,
            'affine': False
        }
        self._set_norm_kwargs = {}
        self._set_feature_block_kwargs = {}
        self._set_relation_block_kwargs = {}
        self._use_rnn_norm = False

    def get_set_kwargs(self):
        config = {}
        config['norm_block'] = dict(self._set_norm_kwargs)
        config['feature_block'] =  dict(self._set_feature_block_kwargs)
        config['relation_block'] =  dict(self._set_relation_block_kwargs)
        config['use_input_norm'] = self.use_input_norm
        config['use_rnn_norm'] = self._use_rnn_norm
        for values in config.values():
            if values is not None:
                if isinstance(values,dict) and 'self' in values:
                    del values['self']

        return config
        
    def set_norm_block(self, norm_class=None, affine=None, momentum=None):
        self._set_norm_kwargs = locals()
        self._norm_config[
            'norm_class'] = norm_class or self._norm_config['norm_class']
        self._norm_config['affine'] = affine or self._norm_config['affine']
        self._norm_config[
            'momentum'] = momentum or self._norm_config['momentum']

    def set_feature_block(self, stack_cnn_class=None, **config):
        self._set_feature_block_kwargs = locals()
        self._feature_block_config.update(config)
        self._use_feature_block = self._feature_block_config['num_layers'] != 0
        self._stack_cnn_class = stack_cnn_class or self._stack_cnn_class

    def set_relation_block(self,out_channels=None,rnn_type=None,
                           filter_num=None,filter_hidden=None,
                           use_first_filter=None,use_second_filter=None,
                           use_common_filter=None,output_act=None,
                           use_norm=None,**config):
        self._set_relation_block_kwargs = locals()
        self._output_act = output_act or self._output_act
        self._rnn_type = rnn_type or self._rnn_type
        self._out_channels = out_channels or self._out_channels
        self._filter_num = filter_num or self._filter_num
        self._filter_hidden = filter_hidden or self._filter_hidden

        if use_norm is not None:
            self._use_rnn_norm = use_norm

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
        norm_input_block = None
        norm_rnn_class = None
        norm_class = generate_norm_class(
            self._norm_config['norm_class'],
            affine=self._norm_config['affine'],
            momentum=self._norm_config['momentum'])
        if self.use_input_norm:
            norm_input_block = norm_class(in_channels)

        if self._use_feature_block:
            feature_block = STACK_CNN_CLASS[self._stack_cnn_class](
                in_channels,
                norm_class=norm_class,
                **self._feature_block_config)
            in_channels = feature_block.out_channels

        if self._use_rnn_norm:
            norm_rnn_class = norm_class
            
        if self._rnn_type in RNN_TYPES:
            rnn_class = RNN_TYPES[self._rnn_type]
            relation_block = rnn_class(in_channels,
                                       output_act=self._output_act,
                                       out_channels=self._out_channels,
                                       norm_class=norm_rnn_class,
                                       **self._relation_block_config)

        elif self._rnn_type == 'FilteredRNN':
            relation_block = FilteredRNN(in_channels,
                                         output_act=self._output_act,
                                         out_channels=self._out_channels,
                                         norm_class=norm_rnn_class,
                                         **self._relation_block_config)

        elif self._rnn_type == 'HierRNN':
            builder = HierRNNBuilder(in_channels,
                                     output_act='sigmoid',
                                     norm_class=norm_rnn_class,
                                     **self._relation_block_config)
            builder.set_filter_place(first=self._use_first_filter,
                                     second=self._use_second_filter,
                                     common=self._use_common_filter)

            builder.set_filter_setting(hidden_size=self._filter_hidden,
                                       num_layers=self._filter_num)
            relation_block = builder.build()
        else:
            raise Exception("{} is not supported".format(self._rnn_type))

        model = SeqAnnModel(feature_block, relation_block, norm_input_block)
        return model


def get_model(config,
              model_weights_path=None,
              frozen_names=None,
              save_distribution=False):
    builder = SeqAnnBuilder()
    if isinstance(config, str):
        config = read_json(config)

    builder.use_input_norm = config['use_input_norm']
    builder.set_norm_block(**config['norm_config'])
    builder.set_feature_block(**config['feature_block_config'])
    builder.set_relation_block(**config['relation_block_config'])
    model = builder.build()
    model.save_distribution = save_distribution

    if model_weights_path is not None:
        print("Load model weights from {}".format(model_weights_path))
        weight = torch.load(model_weights_path,map_location='cpu')
        model.load_state_dict(weight, strict=True)

    frozen_names = frozen_names or []
    for name in frozen_names:
        print("Freeze {}".format(name))
        layer = getattr(model, name)
        for param in layer.named_parameters():
            param[1].requires_grad = False

    return model.cuda()
