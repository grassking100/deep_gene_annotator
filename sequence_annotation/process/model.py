from ..utils.utils import read_json
from .customized_layer import BasicModel, generate_norm_class
from .relation_block import RelationBlockBuilder
from .feature_block import FeatureBlockBuilder


class SeqAnnModel(BasicModel):
    def __init__(self,norm_input_block, feature_block, relation_block):
        super().__init__()
        self._nb = norm_input_block
        self._fb = feature_block
        self._rb = relation_block
        self._in_channels = self._fb.in_channels
        self._out_channels = self._rb.out_channels
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config['norm_input_block'] = self._nb.get_config()
        config['feature_block'] = self._fb.get_config()
        config['relation_block'] = self._rb.get_config()
        return config

    def forward(self, features, lengths, answers=None):
        # shape : N,C,L
        features = self._nb(features, lengths)
        features, lengths = self._fb(features, lengths)
        self._update_distribution(self._fb.saved_distribution)
        result = self._rb(features, lengths, answers=answers)
        self._update_distribution(self._rb.saved_distribution)
        self._update_distribution(result, key='last')
        return result, lengths


class SeqAnnBuilder:
    def __init__(self):
        self._in_channels = 4
        self._norm_config = {}
        self._fb_config = {}
        self._use_rnn_norm = False
        self._rb_config = {}
        
    def set_norm_block(self, **config):
        self._norm_config = config

    def set_feature_block(self, **config):
        self._fb_config = config

    def set_relation_block(self,use_norm=False, **config):
        self._use_rnn_norm = use_norm    
        self._rb_config = config

    def build(self):
        in_channels = self._in_channels
        norm_class = generate_norm_class(**self._norm_config)
        norm_input_block = norm_class(in_channels)
        norm_rnn_class = None
        if self._use_rnn_norm:
            norm_rnn_class = norm_class

        fb_builder = FeatureBlockBuilder()
        fb_builder.set_basic_settings(in_channels,norm_class=norm_class,**self._fb_config)
        feature_block = fb_builder.build()

        rb_builder = RelationBlockBuilder()
        rb_builder.set_basic_settings(feature_block.out_channels,
                                      norm_class=norm_rnn_class,**self._rb_config)
        relation_block = rb_builder.build()

        model = SeqAnnModel(norm_input_block, feature_block, relation_block)
        return model


def create_model(settings,weights_path=None,frozen_names=None,save_distribution=False):
    builder = SeqAnnBuilder()
    if isinstance(settings, str):
        settings = read_json(settings)

    builder.set_norm_block(**settings['norm_kwargs'])
    builder.set_feature_block(**settings['feature_block_kwargs'])
    builder.set_relation_block(**settings['relation_block_kwargs'])
    model = builder.build()
    model.save_distribution = save_distribution
    if weights_path is not None:
        model.load(weights_path)

    frozen_names = frozen_names or []
    for name in frozen_names:
        print("Freeze {}".format(name))
        layer = getattr(model, name)
        for param in layer.named_parameters():
            param[1].requires_grad = False

    return model.cuda()
