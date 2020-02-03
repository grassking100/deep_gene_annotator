import os
import sys
import json
import torch
import deepdish as dd
import pandas as pd
from numpy import median
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP,read_json
from sequence_annotation.utils.utils import CONSTANT_LIST,CONSTANT_DICT
from sequence_annotation.utils.seq_converter import DNA_CODES
from sequence_annotation.genome_handler.select_data import select_data as _select_data
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.process.executor import ExecutorBuilder
from sequence_annotation.process.model import SeqAnnBuilder
from sequence_annotation.process.utils import get_name_parameter

SIMPLIFY_MAP = CONSTANT_DICT({'exon':['exon'],'intron':['intron'],'other':['other']})
BASIC_COLOR_SETTING = CONSTANT_DICT({'other':'blue','exon':'red','intron':'yellow'})

def copy_path(root,path):
    command = 'cp -t {} {}'.format(root,path)
    os.system(command)

def select_data(fasta_path,ann_seqs_path,id_paths,**kwargs):
    id_list = []
    for id_path in id_paths:
        id_list.append(list(pd.read_csv(id_path,header=None)[0]))

    data = _select_data(fasta_path,ann_seqs_path,id_list,
                        simplify_map=SIMPLIFY_MAP,gene_map=BASIC_GENE_MAP,
                        codes=DNA_CODES,**kwargs)
    
    return data[0]

def load_data(path):
    data = dd.io.load(path)
    if isinstance(data[1],dict):
        data = data[0],AnnSeqContainer().from_dict(data[1])
    return data

def get_executor(model,optim_type=None,use_native=True,
                 set_loss=True,set_optimizer=True,
                 label_num=None,predict_label_num=None,
                 answer_label_num=None,output_label_num=None,
                 grad_clip=None,grad_norm=None,
                 learning_rate=None,reduce_lr_on_plateau=False,
                 gamma=None,intron_coef=None,other_coef=None,
                 target_weight_decay=None,weight_decay_name=None,
                 executor_weights_path=None,**kwargs):

    if 'use_naive' in kwargs:
        use_native = kwargs['use_naive']
        del kwargs['use_naive']
    builder = ExecutorBuilder(use_native=use_native,label_num=label_num,
                              predict_label_num=predict_label_num,
                              answer_label_num=answer_label_num,
                              output_label_num=output_label_num,
                              grad_clip=grad_clip,grad_norm=grad_norm)

    target_weight_decay = target_weight_decay or []
    weight_decay_name = weight_decay_name or []
    params = model.parameters()
    if len(target_weight_decay) == len(weight_decay_name):
        params = []
        special_names = []
        for name,weight_decay_ in zip(weight_decay_name,target_weight_decay):
            returned_names,parameters = get_name_parameter(model,[name])
            special_names += returned_names
            params += [{'params':parameters,'weight_decay':weight_decay_}]
        default_parameters = []
        for name_,parameter in model.named_parameters():
            if name_ not in special_names:
                default_parameters.append(parameter)
        params += [{'params':default_parameters}]
    else:
        raise Exception("Different number between target_weight_decay and weight_decay_name")
    
    if set_optimizer:
        builder.set_optimizer(params,optim_type,learning_rate=learning_rate,
                              reduce_lr_on_plateau=reduce_lr_on_plateau,**kwargs)

    if set_loss:
        builder.set_loss(gamma=gamma,intron_coef=intron_coef,other_coef=other_coef)
        
    executor = builder.build(executor_weights_path=executor_weights_path)
    return executor

def get_model(config,model_weights_path=None,frozen_names=None):
    builder = SeqAnnBuilder()
    if isinstance(config,str):
        config = read_json(config)
    builder.config = config
    model = builder.build().cuda()
    if model_weights_path is not None:
        weight = torch.load(model_weights_path)
        model.load_state_dict(weight,strict=True)
        
    frozen_names = frozen_names or []
    for name in frozen_names:
        print("Freeze {}".format(name))
        layer = getattr(model,name)
        for param in layer.named_parameters():
            param[1].requires_grad = False
    return model

def get_model_executor(model_config_path,executor_config_path,
                       model_weights_path=None,frozen_names=None,
                       save_distribution=False):
    #Create model
    model = get_model(model_config_path,model_weights_path=model_weights_path,
                      frozen_names=frozen_names)
    model.save_distribution = save_distribution
    #Create executor
    executor_config = read_json(executor_config_path)
    executor = get_executor(model,**executor_config)
    return model,executor
