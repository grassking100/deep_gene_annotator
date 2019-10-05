import os
import sys
from argparse import ArgumentParser
import pandas as pd
import json
#Load library
import torch
sys.path.append("/home/sequence_annotation")
from sequence_annotation.genome_handler.load_data import load_data as _load_data
from sequence_annotation.process.loss import SeqAnnLoss, FocalLoss, MixedLoss, LabelLoss,SiteLoss
from sequence_annotation.process.executor import BasicExecutor, GANExecutor
from sequence_annotation.process.inference import seq_ann_inference, seq_ann_reverse_inference, basic_inference
from sequence_annotation.process.model import SeqAnnBuilder

BEFORE_MIX_SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron','alt_accept','alt_donor'],'other':['other']}
SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron'],'other':['other']}
GENE_MAP = {'gene':['exon','intron'],'other':['other']}
BASIC_COLOR_SETTING={'other':'blue','exon':'red','intron':'yellow'}
ANN_TYPES = ['exon','intron','other']
#CHANNEL_ORDER = ANN_TYPES + ['TSSs', 'ca_sites', 'donor_sites','accept_sites']

def load_data(fasta_path,ann_seqs_path,id_paths,**kwargs):
    id_list = []
    for id_path in id_paths:
        id_list.append(list(pd.read_csv(id_path,header=None)[0]))

    if 'max_len' in kwargs and kwargs['max_len'] < 0:
        kwargs['max_len'] = None

    data = _load_data(fasta_path,ann_seqs_path,id_list,simplify_map=SIMPLIFY_MAP,
                      before_mix_simplify_map=BEFORE_MIX_SIMPLIFY_MAP,
                      gene_map=GENE_MAP,**kwargs)
    
    return data

def get_executor(model,use_naive=True,use_discrim=False,set_loss=True,set_optimizer=True,
                 learning_rate=None,disrim_learning_rate=None,intron_coef=None,other_coef=None,
                 nontranscript_coef=None,gamma=None,transcript_answer_mask=False,
                 transcript_output_mask=True,mean_by_mask=False,frozed_names=None,
                 weight_decay=None,site_mask_method=None,label_num=None,
                 predict_label_num=None,answer_label_num=None,output_label_num=None,**kwargs):

    if learning_rate is None:
        learning_rate =  1e-3
    if disrim_learning_rate is None:
        disrim_learning_rate = 1e-3
        
    weight_decay = weight_decay or 0

    if use_naive:
        output_label_num = predict_label_num = answer_label_num = label_num or 3
    else:
        predict_label_num = predict_label_num or 2
        answer_label_num = answer_label_num or 3
        output_label_num = output_label_num or 3

    print(predict_label_num,answer_label_num,output_label_num)
    if use_discrim:
        executor = GANExecutor()
    else:
        executor = BasicExecutor()
    if use_naive:
        executor.inference = basic_inference(output_label_num)
    else:
        executor.inference = seq_ann_inference
    if not use_naive and use_discrim:
        executor.reverse_inference = seq_ann_reverse_inference

    if set_optimizer:
        if use_discrim:
            executor.optimizer = (torch.optim.Adam(filter(lambda p: p.requires_grad, model.gan.parameters()),
                                                   lr=learning_rate,weight_decay=weight_decay),
                                  torch.optim.Adam(filter(lambda p: p.requires_grad, model.discrim.parameters()),
                                                   lr=disrim_learning_rate,weight_decay=weight_decay))
        else:
            executor.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                                  lr=learning_rate,weight_decay=weight_decay)
    
    if set_loss:
        if use_naive:
            loss = FocalLoss(gamma)
        else:    
            loss = SeqAnnLoss(intron_coef=intron_coef,other_coef=other_coef,
                              nontranscript_coef=nontranscript_coef,
                              transcript_answer_mask=transcript_answer_mask,
                              transcript_output_mask=transcript_output_mask,
                              mean_by_mask=mean_by_mask)
        label_loss = LabelLoss(loss)
        label_loss.predict_inference = basic_inference(predict_label_num)
        label_loss.answer_inference = basic_inference(answer_label_num)
    
        if site_mask_method is not None:
            site_loss = SiteLoss(FocalLoss(gamma))
            site_loss.output_inference = basic_inference(output_label_num,before=False)
            site_loss.answer_inference = basic_inference(answer_label_num,before=False)
            executor.loss = MixedLoss(label_loss=label_loss,site_loss=site_loss,site_mask_method=site_mask_method)
        else:
            executor.loss = label_loss
    else:
        executor.loss = None

    return executor

def get_model(path_or_json,model_weights_path=None,frozen_names=None):
    builder = SeqAnnBuilder()
    if isinstance(path_or_json,str):
        with open(path_or_json,"r") as fp:
            config = json.load(fp)
            builder.config = config
    else:
        builder.config = path_or_json
    model = builder.build().cuda()
    
    if model_weights_path is not None:
        weight = torch.load(model_weights_path)
        model.load_state_dict(weight,strict=False)
        
    frozen_names = frozen_names or []
    for name in frozen_names:
        print("Freeze {}".format(name))
        layer = getattr(model,name)
        for param in layer.named_parameters():
            param[1].requires_grad = False

    return model
