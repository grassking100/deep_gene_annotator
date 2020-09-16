from .rnn import RNN_CLASSES
from .hier_rnn import HierRNNBuilder

class RelationBlockBuilder:
    IN_CHANNELS = 4
    OUT_CHANNELS = 3
    HIDDEN_SIZES = 16
    OUTPUT_ACT = 'softmax'
    RNN_TYPE = 'ProjectedRNN'
    #SITE_NUM = 4
    KWARGS = {}
    def __init__(self):
        self._in_channels = self._out_channels = self._hidden_size = None
        self._output_act = self._rnn_type = self._kwargs = None
        self.reset()

    def set_basic_settings(self,in_channels=None, hidden_size=None, 
                           out_channels=None, rnn_type=None, output_act=None,
                           **kwargs):
        kwargs_ = dict(kwargs)
        for key,value in kwargs.items():
            if value is None:
                del kwargs_[key]
        if rnn_type not in list(RNN_CLASSES.keys()) + ['HierRNN', None]:
            raise Exception("{} is not supported".format(rnn_type))
        self._in_channels = in_channels or self.IN_CHANNELS
        self._hidden_size = hidden_size or self.HIDDEN_SIZES
        self._out_channels = out_channels or self.OUT_CHANNELS
        self._rnn_type = rnn_type or self.RNN_TYPE
        self._output_act = output_act or self.OUTPUT_ACT
        self._kwargs = kwargs_ or dict(self.KWARGS)
        
    def reset(self):
        self._in_channels = self.IN_CHANNELS
        self._out_channels = self.OUT_CHANNELS
        self._hidden_size = self.HIDDEN_SIZES
        self._rnn_type = self.RNN_TYPE
        self._output_act = self.OUTPUT_ACT
        self._kwargs = dict(self.KWARGS)
        
    def build(self):
        if self._rnn_type in RNN_CLASSES:
            rnn_class = RNN_CLASSES[self._rnn_type]
            relation_block = rnn_class(self._in_channels,self._hidden_size,self._out_channels,
                                       output_act=self._output_act,**self._kwargs)
        else:
            hier_kwargs = dict(self._kwargs)
            
            builder = HierRNNBuilder(self._in_channels,self._hidden_size,**hier_kwargs)
            relation_block = builder.build()
        return relation_block
