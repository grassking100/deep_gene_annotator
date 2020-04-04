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
from sequence_annotation.utils.seq_converter import DNA_CODES
from sequence_annotation.genome_handler.select_data import select_data as _select_data
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.process.utils import BASIC_SIMPLIFY_MAP

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
                        simplify_map=BASIC_SIMPLIFY_MAP,gene_map=BASIC_GENE_MAP,
                        codes=DNA_CODES,**kwargs)
    return data

def load_data(path):
    data = dd.io.load(path)
    data = data[0],AnnSeqContainer().from_dict(data[1])
    return data
