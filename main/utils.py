import os
import sys
import torch
import random
import deepdish as dd
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import BASIC_GENE_MAP,read_json
from sequence_annotation.utils.utils import CONSTANT_DICT
from sequence_annotation.utils.seq_converter import DNA_CODES
from sequence_annotation.genome_handler.select_data import select_data as _select_data
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.process.executor import ExecutorBuilder
from sequence_annotation.process.model import SeqAnnBuilder
from sequence_annotation.process.utils import get_name_parameter

SIMPLIFY_MAP = CONSTANT_DICT({'exon':['exon'],'intron':['intron'],'other':['other']})
BASIC_COLOR_SETTING = CONSTANT_DICT({'other':'blue','exon':'red','intron':'yellow'})

def backend_deterministic(deterministic):
    if deterministic:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
    torch.backends.cudnn.enabled = not deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

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

def get_params(model,target_weight_decay=None,weight_decay_name=None):
    target_weight_decay = target_weight_decay or []
    weight_decay_name = weight_decay_name or []
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
        return params
    else:
        raise Exception("Different number between target_weight_decay and weight_decay_name")

def get_executor(model,config,executor_weights_path=None):
    if isinstance(config,str):
        config = read_json(config)

    builder = ExecutorBuilder(config['use_native'])
    params = get_params(model,**config['weight_decay_config'])
    builder.set_optimizer(**config['optim_config'])
    builder.set_loss(**config['loss_config'])
    executor = builder.build(params,executor_weights_path=executor_weights_path)
    return executor

def get_model(config,model_weights_path=None,frozen_names=None,save_distribution=False):
    builder = SeqAnnBuilder()
    if isinstance(config,str):
        config = read_json(config)

    builder.set_feature_block(**config['feature_block_config'])
    builder.set_relation_block(**config['relation_block_config'])
    model = builder.build()
    model.save_distribution = save_distribution
    if model_weights_path is not None:
        weight = torch.load(model_weights_path)
        model.load_state_dict(weight,strict=True)
        
    frozen_names = frozen_names or []
    for name in frozen_names:
        print("Freeze {}".format(name))
        layer = getattr(model,name)
        for param in layer.named_parameters():
            param[1].requires_grad = False

    return model.cuda()

def get_model_executor(model_config_path,executor_config_path,
                       model_weights_path=None,frozen_names=None,
                       save_distribution=False):
    #Create model
    model = get_model(model_config_path,
                      model_weights_path=model_weights_path,
                      frozen_names=frozen_names,
                      save_distribution=save_distribution)
    #Create executor
    executor = get_executor(model,executor_config_path)
    return model,executor
