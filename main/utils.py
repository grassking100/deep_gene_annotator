import os
import sys
import torch
import random
import deepdish as dd
import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import BASIC_GENE_MAP
from sequence_annotation.utils.utils import CONSTANT_DICT
from sequence_annotation.utils.seq_converter import DNA_CODES
from sequence_annotation.genome_handler.select_data import select_data as _select_data
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.process.executor import get_executor
from sequence_annotation.process.model import get_model

SIMPLIFY_MAP = CONSTANT_DICT({'exon':['exon'],'intron':['intron'],'other':['other']})
BASIC_COLOR_SETTING = CONSTANT_DICT({'other':'blue','exon':'red','intron':'yellow'})

def backend_deterministic(deterministic,benchmark=False):
    if deterministic:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
    torch.backends.cudnn.enabled = not deterministic
    if deterministic:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

def select_data(fasta_path,ann_seqs_path,id_path,**kwargs):
    ids = list(pd.read_csv(id_path,header=None)[0])
    data = _select_data(fasta_path,ann_seqs_path,ids,
                        simplify_map=SIMPLIFY_MAP,gene_map=BASIC_GENE_MAP,
                        codes=DNA_CODES,**kwargs)
    return data

def load_data(path):
    data = dd.io.load(path)
    data = data[0],AnnSeqContainer().from_dict(data[1])
    return data

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
