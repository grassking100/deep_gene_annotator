import math
import sys,os
from collections import OrderedDict
sys.path.append(os.path.dirname(__file__)+"/..")
from sequence_annotation.process.model import SeqAnnBuilder
from sequence_annotation.process.executor import ExecutorBuilder
from sequence_annotation.process.optuna import IModelExecutorCreator,get_discrete_uniform,get_bool_choices,get_choices

class ModelExecutorCreator(IModelExecutorCreator):
    """Creator to create model and executor by trial parameters"""
    def __init__(self,clip_grad_norm=None,has_cnn=True,grad_norm_type=None,
                 lr_scheduler_patience=None,clip_grad_value_by_hook=None,dropout=None,**kwargs):
        self._dropout = dropout
        self._grad_norm_type = grad_norm_type or 'inf'
        self._clip_grad_norm = clip_grad_norm
        self._clip_grad_value_by_hook = clip_grad_value_by_hook
        self._has_cnn = has_cnn
        self._lr_scheduler_patience = lr_scheduler_patience
        self._cnn_config_dict = {'cnn_num':'num_layers','cnn_out':'out_channels','kernel_size':'kernel_size'}
        self._rnn_config_dict = {'rnn_hidden':'hidden_size','rnn_num':'num_layers'}
        self._all_config_dict = dict(self._cnn_config_dict)
        self._all_config_dict.update(self._rnn_config_dict)
        self._each_space_sizes = self.get_each_space_size()
        self._space_size = self._get_space_size()
        self._all_configs = self.create_all_configs()
 
    @property
    def space_size(self):
        return self._space_size

    def create_default_hyperparameters(self):
        config = {}
        #CNN
        if self._has_cnn:
            config['kernel_size'] = {'lb':65,'ub':321,'step':128,'value':None}#Num: 3
            config['cnn_out'] = {'lb':8,'ub':16,'step':4,'value':None}#Num: 3
            config['cnn_num'] = {'lb':8,'ub':16,'step':4,'value':None}#Num: 3
        else:
            config['cnn_num'] = {'value':0}
        #RNN
        config['relation_type'] = {'options':['basic','basic_hier','hier'],'value':None}#Num: 3
        config['rnn_num'] = {'lb':1,'ub':3,'value':None}#Num: 3
        config['rnn_hidden'] = {'lb':64,'ub':128,'step':32,'value':None}#Num: 3
        config['is_rnn_filter'] = {'value':False}
        config['rnn_type'] = {'value':'GRU'}
        #Executor
        config['optimizer_type'] = {'value':'Adam'}
        config['learning_rate'] = {'value':1e-3}
        if self._clip_grad_norm is not None:
            config['clip_grad_norm'] = {'value':self._clip_grad_norm}
            config['grad_norm_type'] = {'value':float(self._grad_norm_type)}

        ordered_config = OrderedDict()
        for key in sorted(list(config.keys())):
            ordered_config[key] = config[key]
        return ordered_config

    def _create_model_builder(self):
        model_builder = SeqAnnBuilder()
        model_builder.set_feature_block(
            customized_init='kaiming_uniform_cnn_init',
            padding_handle='same',
            dropout=self._dropout
        )
        model_builder.set_relation_block(
            customized_cnn_init='xavier_uniform_cnn_init',
            customized_rnn_init='in_xav_bias_zero_gru_init',
            dropout=self._dropout
        )
        return model_builder        

    def _create_executor_builder(self):
        builder = ExecutorBuilder()
        builder.set_optimizer(clip_grad_value_by_hook=self._clip_grad_value_by_hook)
        if self._lr_scheduler_patience is not None:
            builder.set_lr_scheduler(patience=self._lr_scheduler_patience,
                                     threshold=0,factor=0.5,
                                     use_lr_scheduler=True)
        return builder
    
    def _set_hyperparameters_from_trial(self,trial,hyper):
        for name, value in trial.params.items():
            if name in ['cnn_num','kernel_size','rnn_num','cnn_out','rnn_hidden']:
                value = int(value)
            print("Set trial.param's {} to {}".format(name,value))
            
            hyper[name] = {'value':value}
        for name, value in trial.system_attrs.items():
            print("Set trial.system_attrs's {} to {}".format(name,value))
            hyper[name] = {'value':value}
            
        for name, value in trial.user_attrs.items():
            print("Set trial.user_attrs's {} to {}".format(name,value))
            hyper[name] = {'value':value}
    
    def _set_builder(self,hyper,model_builder,executor_builder):
        for key,set_key in self._cnn_config_dict.items():
            if key in hyper:
                value = hyper[key]['value']
                model_builder.set_feature_block(**{set_key:value})

        for key,set_key in self._rnn_config_dict.items():
            if key in hyper:
                value = hyper[key]['value']
                model_builder.set_relation_block(**{set_key:value})

        is_filter = hyper['is_rnn_filter']['value']
        relation_type = hyper['relation_type']['value']
        rnn_type = hyper['rnn_type']['value']
        is_gru = rnn_type == 'GRU'
        if relation_type == 'hier':
            model_builder.set_relation_block(rnn_type='HierRNN',is_gru=is_gru,
                                             use_first_filter=is_filter,
                                             use_second_filter=is_filter)
            executor_builder.use_native = False
        elif relation_type=='basic' or relation_type=='basic_hier':
            if is_filter:
                rnn_type = 'FilteredRNN'
            else:
                rnn_type = 'ProjectedRNN'

            if relation_type == 'basic':
                output_act = 'softmax'
                out_channels = 3
            else:
                output_act = 'sigmoid'
                out_channels = 2

            model_builder.set_relation_block(rnn_type=rnn_type,is_gru=is_gru,
                                             out_channels=out_channels,output_act=output_act)

            executor_builder.use_native = relation_type == 'basic'
        else:
            raise Exception("Unknown relation block type {}".format(relation_type))

        clip_grad_norm = hyper['clip_grad_norm']['value'] if 'clip_grad_norm' in hyper else None
        grad_norm_type = hyper['grad_norm_type']['value'] if 'grad_norm_type' in hyper else None
        executor_builder.set_optimizer(hyper['optimizer_type']['value'],
                                       learning_rate=hyper['learning_rate']['value'],
                                       clip_grad_norm=clip_grad_norm,
                                       grad_norm_type=grad_norm_type)
    
    def get_trial_config(self,trial,set_by_trial_config=False):
        self._index =trial.number
        hyper = self.create_default_hyperparameters()
        if set_by_trial_config:
            self._set_hyperparameters_from_trial(trial,hyper)
        
        for key,config in hyper.items():
            if config['value'] is None:
                if 'options' in config:
                    hyper[key]['value'] = get_choices(trial,key,config['options'])
                elif 'ub' in config:
                    hyper[key]['value'] = int(get_discrete_uniform(trial,key,**config))
                else:
                    hyper[key]['value'] = get_bool_choices(trial,key)
        return hyper
    
    def get_each_space_size(self):
        hyper = self.create_default_hyperparameters()
        space_size = {}
        for key,config in hyper.items():
            if config['value'] is None:
                #It is discrete int space
                if 'lb' in config:
                    step = config['step'] if 'step' in config else 1
                    space_size[key] = math.floor((config['ub'] - config['lb'])/step)+1
                #It is categorical space
                elif 'options' in config:
                    space_size[key] = len(config['options'])
                #It is bool space
                else:
                    space_size[key] = 2
            else:
                space_size[key] = 1
        return space_size
    
    def _get_space_size(self):
        space_size = 1
        for value in self._each_space_sizes.values():
            space_size *= value
        return space_size
    
    def _get_value(self,config,index):
        if config['value'] is None:
            if 'lb' in config:
                step = config['step'] if 'step' in config else 1
                value = config['lb'] + step*index
            #It is categorical space
            elif 'options' in config:
                value = config['options'][index]
            #It is bool space
            else:
                value = [False,True][index]
        else:    
            value = config['value']
        return value
        
    def create_all_configs(self):
        hyper = self.create_default_hyperparameters()
        space_size = self.space_size
        all_configs = []
        for id_ in range(space_size):
            unique_config = {}
            for key,config in hyper.items():
                each_space_size =  self._each_space_sizes[key]
                index = math.floor(id_%(each_space_size))
                id_ = math.floor(id_/(each_space_size))
                unique_config[key] = {'value':self._get_value(config,index)}
            all_configs.append(unique_config)
        return all_configs
        
    def _create(self,config):
        model_builder = self._create_model_builder()
        executor_builder = self._create_executor_builder()
        self._set_builder(config,model_builder,executor_builder)
        model = model_builder.build().cuda()
        executor = executor_builder.build(model.parameters())
        return {
            'model':model,'executor':executor
        }
        
    def _create_by_grid_search(self,trial):
        if trial.number >= self.space_size:
            raise Exception("The index is exceed space size")
        config = self._all_configs[trial.number]
        for key,size in self._each_space_sizes.items():
            if size > 1:
                trial.set_user_attr(key,config[key]['value'])
                
        return self._create(config)

    def create_by_trial(self,trial,set_by_trial_config=False,set_by_grid_search=False):
        trial.set_user_attr("set_by_grid_search",set_by_grid_search)
        if set_by_grid_search:
            return self._create_by_grid_search(trial)
        else:
            hyper = self.get_trial_config(trial,set_by_trial_config=set_by_trial_config)
            return self._create(hyper)

    def create_max(self):
        hyper = self.create_default_hyperparameters()
        for key,config in hyper.items():
            if 'ub' in config:
                hyper[key]['value'] = config['ub']
                
        hyper['relation_type']['value'] = 'hier'
        return self._create(hyper)
