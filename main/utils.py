import os
import sys
from argparse import ArgumentParser
import pandas as pd
import json
#Load library
import torch
sys.path.append("/home/sequence_annotation")
from sequence_annotation.genome_handler.load_data import load_data as _load_data
from sequence_annotation.pytorch.loss import SeqAnnLoss, FocalLoss
from sequence_annotation.pytorch.executor import BasicExecutor, GANExecutor
from sequence_annotation.pytorch.model import seq_ann_inference, SeqAnnBuilder
from sequence_annotation.pytorch.model import seq_ann_reverse_inference

BEFORE_MIX_SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron','alt_accept','alt_donor'],'other':['other']}
SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron'],'other':['other']}
GENE_MAP = {'gene':['exon','intron'],'other':['other']}
BASIC_COLOR_SETTING={'other':'blue','exon':'red','intron':'yellow'}

def load_data(fasta_path,ann_seqs_path,train_id_path,val_id_path=None,min_len=None,max_len=None):
    train_ids = list(pd.read_csv(train_id_path,header=None)[0])
    val_ids = []
    val_data = None
    if val_id_path is not None:
        val_ids = list(pd.read_csv(val_id_path,header=None)[0])
    data = _load_data(fasta_path,ann_seqs_path,[train_ids,val_ids],
                      min_len=min_len,max_len=max_len,simplify_map=SIMPLIFY_MAP,
                      before_mix_simplify_map=BEFORE_MIX_SIMPLIFY_MAP)
    return data

def get_executor(model,use_naive=True,use_discrim=False,learning_rate=None,disrim_learning_rate=None,
                 intron_coef=None,other_coef=None,nontranscript_coef=None,gamma=None,
                 transcript_answer_mask=False,transcript_output_mask=True,mean_by_mask=False,**kwargs):
        
    learning_rate = learning_rate or 1e-3
    disrim_learning_rate = disrim_learning_rate or 1e-3

    if use_discrim:
        executor = GANExecutor()
        executor.reverse_inference = seq_ann_reverse_inference
        executor.optimizer = (torch.optim.Adam(model.gan.parameters(),
                                               lr=learning_rate),
                              torch.optim.Adam(model.discrim.parameters(),
                                               lr=disrim_learning_rate))
    else:
        executor = BasicExecutor()
        executor.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=learning_rate)

    if use_naive:  
        executor.loss = FocalLoss(gamma)
    else:    
        executor.loss = SeqAnnLoss(intron_coef=intron_coef,other_coef=other_coef,
                                   nontranscript_coef=nontranscript_coef,
                                   transcript_answer_mask=transcript_answer_mask,
                                   transcript_output_mask=transcript_output_mask,
                                   mean_by_mask=mean_by_mask)
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
