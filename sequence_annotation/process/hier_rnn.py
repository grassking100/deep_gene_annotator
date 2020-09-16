import torch
from .customized_layer import BasicModel
from .rnn import ProjectedRNN

class HierRNN(BasicModel):
    def __init__(self, rnn_0, rnn_1, hier_option=None):
        super().__init__()
        self._hier_option = hier_option or 'before_filter'
        if self._hier_option not in ['before_filter', 'after_filter', 'independent']:
            raise Exception("Wrong hier_option")
        self._rnn_0 = rnn_0
        self._rnn_1 = rnn_1
        self._in_channels = rnn_0.in_channels
        self._out_channels = self._rnn_0.out_channels + self._rnn_1.out_channels
        self.reset_parameters()

    def get_config(self):
        config = super().get_config()
        config['hier_option'] = self._hier_option
        config['rnn_0'] = self._rnn_0.get_config()
        config['rnn_1'] = self._rnn_1.get_config()
        return config

    def forward(self, x, lengths, answers=None, **kwargs):
        result_0 = self._rnn_0(x, lengths)
        if self._hier_option == 'independent':
            result_1 = self._rnn_1(x, lengths)
        else:
            gated_x = x * result_0
            self._update_distribution(gated_x, key='gated_x')
            if self._hier_option == 'before_filter':
                result_1 = self._rnn_1(gated_x, lengths)
            else:
                result_1 = self._rnn_1(x, lengths, target_feature=gated_x)
        result = torch.cat([result_0, result_1], 1)
        self._update_distribution(self._rnn_0.saved_distribution)
        self._update_distribution(self._rnn_1.saved_distribution)
        self._update_distribution(result, key='gated_stack_result')
        return result


class HierRNNBuilder:
    def __init__(self, in_channels, hidden_size, hier_option=None, **kwargs):
        self._in_channels = in_channels
        self._hidden_size = hidden_size
        self._kwargs = kwargs
        self._hier_option = hier_option

    def _create_rnn(self, name):
        rnn = ProjectedRNN(self._in_channels,self._hidden_size,out_channels=1,
                           name=name,output_act='sigmoid',**self._kwargs)
        return rnn

    def build(self):
        rnn_0 = self._create_rnn('first')
        rnn_1 = self._create_rnn('second')
        rnn = HierRNN(rnn_0, rnn_1, hier_option=self._hier_option)
        return rnn
