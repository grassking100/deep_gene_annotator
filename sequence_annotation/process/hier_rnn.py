import torch
from .customized_layer import BasicModel
from .rnn import ProjectedRNN
from .filter import RNNFilter,FilteredRNN

class HierRNN(BasicModel):
    def __init__(self,preprocess,rnn_0,rnn_1,hier_option=None):
        super().__init__()
        self.hier_option = hier_option or 'before_filter'
        if self.hier_option not in ['before_filter','after_filter','independent']:
            raise Exception("Wrong hier_option")
        self.preprocess = None
        self.rnn_0 = rnn_0
        self.rnn_1 = rnn_1
        self.in_channels = rnn_0.in_channels
        self.out_channels = self.rnn_0.out_channels + self.rnn_1.out_channels
        self.reset_parameters()
        
    def get_config(self):
        config = super().get_config()
        config['hier_option'] = self.hier_option
        config['rnn_0'] = self.rnn_0.get_config()
        config['rnn_1'] = self.rnn_1.get_config()
        if self.preprocess is not None:
            config['preprocess'] = self.preprocess.get_config()
        return config
        
    def forward(self,x,lengths,answers=None,**kwargs):
        if self.preprocess is not None:
            x = self.preprocess(x,lengths)
            self.update_distribution(x,key='preprocessed_value')

        result_0 = self.rnn_0(x,lengths)
        if self.hier_option == 'independent':
            result_1 = self.rnn_1(x,lengths)
        else:
            gated_x = x*result_0
            self.update_distribution(gated_x,key='gated_x')
            if self.hier_option == 'before_filter':
                result_1 = self.rnn_1(gated_x,lengths)
            else:
                result_1 = self.rnn_1(x,lengths,target_feature=gated_x)
        result = torch.cat([result_0,result_1],1)
        self.update_distribution(self.rnn_0.saved_distribution)
        self.update_distribution(self.rnn_1.saved_distribution)
        self.update_distribution(result,key='gated_stack_result')
        return result

class HierRNNBuilder:
    def __init__(self,in_channels,hier_option=None,**kwargs):
        self._in_channels = in_channels
        self._kwargs = kwargs
        self._hier_option = hier_option
        self._filter_hidden = None
        self._filter_num = None

        self._use_first_filter = False
        self._use_second_filter = False
        self._use_common_filter = False
        
    def set_filter_setting(self,hidden_size=None,num_layers=None):
        self._filter_hidden=hidden_size
        self._filter_num=num_layers
    
    def set_filter_place(self,first=False,second=False,common=False):
        self._use_first_filter=first
        self._use_second_filter=second
        self._use_common_filter=common

    def _create_rnn(self,use_filter,name):
        if use_filter:
            rnn = FilteredRNN(self._in_channels,out_channels=1,name=name,
                              filter_num_layers=self._filter_num,
                              filter_hidden_size=self._filter_hidden,**self._kwargs)
        else:
            rnn = ProjectedRNN(self._in_channels,out_channels=1,name=name,
                              **self._kwargs)  
        return rnn
        
    def _create_rnn_filter(self,name):
        rnn = RNNFilter(self._in_channels,name=name,
                        num_layers=self._filter_num,hidden_size=self._filter_hidden,
                        **self._kwargs)
        return rnn
        
    def build(self):
        rnn_0 = self._create_rnn(self._use_first_filter,'first')
        rnn_1 = self._create_rnn(self._use_second_filter,'second')
        common = None
        if self._use_common_filter:
            common = self._create_rnn_filter('common')
        rnn = HierRNN(common,rnn_0,rnn_1,hier_option=self._hier_option)
        return rnn
