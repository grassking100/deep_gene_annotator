import os
import sys
from argparse import ArgumentParser
import pandas as pd
import json
#Load library
import torch
sys.path.append("/home/sequence_annotation")
from sequence_annotation.genome_handler.load_data import load_data as _load_data
from sequence_annotation.process.loss import SeqAnnLoss, FocalLoss
from sequence_annotation.process.executor import BasicExecutor, GANExecutor
from sequence_annotation.process.inference import seq_ann_inference, seq_ann_reverse_inference
from sequence_annotation.process.model import SeqAnnBuilder

BEFORE_MIX_SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron','alt_accept','alt_donor'],'other':['other']}
SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron'],'other':['other']}
GENE_MAP = {'gene':['exon','intron'],'other':['other']}
BASIC_COLOR_SETTING={'other':'blue','exon':'red','intron':'yellow'}

def load_data(fasta_path,ann_seqs_path,id_paths,min_len=None,max_len=None,ratio=None):
    id_list = []
    for id_path in id_paths:
        id_list.append(list(pd.read_csv(id_path,header=None)[0]))

    if max_len is not None and max_len < 0:
        max_len = None

    data = _load_data(fasta_path,ann_seqs_path,id_list,
                      min_len=min_len,max_len=max_len,simplify_map=SIMPLIFY_MAP,
                      before_mix_simplify_map=BEFORE_MIX_SIMPLIFY_MAP,
                      ratio=ratio)
    return data

def get_executor(model,use_naive=True,use_discrim=False,
                 set_loss=True,set_optimizer=True,
                 learning_rate=None,disrim_learning_rate=None,
                 intron_coef=None,other_coef=None,nontranscript_coef=None,gamma=None,
                 transcript_answer_mask=False,transcript_output_mask=True,mean_by_mask=False,**kwargs):
    if learning_rate is None:
        learning_rate =  1e-3
    if disrim_learning_rate is None:
        disrim_learning_rate = 1e-3

    if use_discrim:
        executor = GANExecutor()
        executor.reverse_inference = seq_ann_reverse_inference
    else:
        executor = BasicExecutor()
            
    if set_optimizer:
        if use_discrim:            
            executor.optimizer = (torch.optim.Adam(model.gan.parameters(),
                                                   lr=learning_rate),
                                  torch.optim.Adam(model.discrim.parameters(),
                                                   lr=disrim_learning_rate))
        else:
            executor.optimizer = torch.optim.Adam(model.parameters(),
                                                  lr=learning_rate)
    
    if set_loss:
        if use_naive:  
            executor.loss = FocalLoss(gamma)
        else:    
            executor.loss = SeqAnnLoss(intron_coef=intron_coef,other_coef=other_coef,
                                       nontranscript_coef=nontranscript_coef,
                                       transcript_answer_mask=transcript_answer_mask,
                                       transcript_output_mask=transcript_output_mask,
                                       mean_by_mask=mean_by_mask)
    else:
        executor.loss = None

    if not use_naive:
        executor.inference = seq_ann_inference

    return executor

def get_model(model_config_path,model_weights_path=None):
    builder = SeqAnnBuilder()
    with open(model_config_path,"r") as fp:
        builder.config = json.load(fp)
    model = builder.build().cuda()
    
    if model_weights_path is not None:
        weight = torch.load(model_weights_path)
        model.load_state_dict(weight)
    return model
