import os
import sys
import warnings
from argparse import ArgumentParser
import pandas as pd
import json
from numpy import median
import torch
from torch import optim
sys.path.append("/home/sequence_annotation")
from sequence_annotation.genome_handler.load_data import load_data as _load_data
from sequence_annotation.process.loss import SeqAnnLoss, FocalLoss, LabelLoss
from sequence_annotation.process.executor import BasicExecutor
from sequence_annotation.process.inference import seq_ann_inference, seq_ann_reverse_inference, basic_inference
from sequence_annotation.process.model import SeqAnnBuilder

BEFORE_MIX_SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron','alt_accept','alt_donor'],'other':['other']}
SIMPLIFY_MAP = {'exon':['exon'],'intron':['intron'],'other':['other']}
GENE_MAP = {'gene':['exon','intron'],'other':['other']}
BASIC_COLOR_SETTING={'other':'blue','exon':'red','intron':'yellow'}
ANN_TYPES = ['exon','intron','other']

def copy_path(root,path):
    command = 'cp -t {} {}'.format(root,path)
    os.system(command)

def select_data_by_length_each_type(fasta,ann_seqs,**kwargs):
    if len(set(ANN_TYPES) - set(ann_seqs.ANN_TYPES)) > 0:
        raise Exception("ANN_TYPES should include {}, but got {}".format(ANN_TYPES,ann_seqs.ANN_TYPES))
    multiple_exon_transcripts = []
    single_exon_transcripts = []
    no_transcripts = []
    #Classify sequence
    for ann_seq in ann_seqs:
        #If it is multiple exon transcript
        if sum(ann_seq.get_ann('intron')) > 0:
            multiple_exon_transcripts.append(ann_seq)
        #If it is single exon transcript
        elif sum(ann_seq.get_ann('exon')) > 0:
            single_exon_transcripts.append(ann_seq)
        #If there is no transcript
        else:
            no_transcripts.append(ann_seq)
            
    selected_seqs = {}
    selected_anns = ann_seqs.copy()
    selected_anns.clean()
    
    for seqs in [multiple_exon_transcripts,single_exon_transcripts,no_transcripts]:
        median_length = median([seq.length for seq in seqs])
        for seq in seqs:
            if seq.length <= median_length:
                selected_seqs[seq.id] = fasta[seq.id]
                selected_anns.add(seq)
    return selected_seqs,selected_anns
    
def load_data(fasta_path,ann_seqs_path,id_paths,select_each_type=False,**kwargs):
    id_list = []
    for id_path in id_paths:
        id_list.append(list(pd.read_csv(id_path,header=None)[0]))

    if 'max_len' in kwargs and kwargs['max_len'] < 0:
        kwargs['max_len'] = None
    
    select_func = None
    if select_each_type:
        select_func = select_data_by_length_each_type

    data = _load_data(fasta_path,ann_seqs_path,id_list,simplify_map=SIMPLIFY_MAP,
                      before_mix_simplify_map=BEFORE_MIX_SIMPLIFY_MAP,
                      gene_map=GENE_MAP,select_func=select_func,**kwargs)
    
    return data

OPTIMIZER_CLASS = {'Adam':optim.Adam,'SGD':optim.SGD,'AdamW':optim.AdamW,'RMSprop':optim.RMSprop}

def optimizer_generator(type_,model,momentum=None,nesterov=False,amsgrad=False,adam_betas=None,**kwargs):
    
    momentum = momentum or 0
    adam_betas = adam_betas or [0.9,0.999]
    if type_ not in OPTIMIZER_CLASS:
        raise Exception("Optimizer should be {}, but got {}".format(OPTIMIZER_CLASS,type_))
        
    filter_ = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = OPTIMIZER_CLASS[type_]
    if type_ == 'AdamW':
        warnings.warn("\n!!!\n\nAdamW's weight decay is implemented differnetly in paper and pytorch 1.3.0, please see https://github.com/pytorch/pytorch/pull/21250#issuecomment-520559993 for more information!\n\n!!!\n")
        
    if optimizer in [optim.Adam,optim.AdamW]:
        if momentum > 0 or nesterov:
            raise
        return optimizer(filter_,amsgrad=amsgrad,betas=adam_betas,**kwargs)
    elif optimizer in [optim.RMSprop]:
        if nesterov or amsgrad:
            raise
        return optimizer(filter_,momentum=momentum,**kwargs)
    else:
        if amsgrad:
            raise
        return optimizer(filter_,momentum=momentum,
                          nesterov=nesterov,**kwargs)

def get_executor(model,optim_type,use_naive=True,set_loss=True,set_optimizer=True,
                 learning_rate=None,intron_coef=None,other_coef=None,
                 nontranscript_coef=None,gamma=None,transcript_answer_mask=True,
                 transcript_output_mask=False,mean_by_mask=False,frozed_names=None,
                 label_num=None,predict_label_num=None,answer_label_num=None,output_label_num=None,
                 grad_clip=None,grad_norm=None,reduce_lr_on_plateau=False,**kwargs):

    learning_rate = learning_rate or 1e-3

    if use_naive:
        output_label_num = predict_label_num = answer_label_num = label_num or 3
    else:
        predict_label_num = predict_label_num or 2
        answer_label_num = answer_label_num or 3
        output_label_num = output_label_num or 3

    executor = BasicExecutor()
        
    executor.grad_clip = grad_clip
    executor.grad_norm = grad_norm
        
    if use_naive:
        executor.inference = basic_inference(output_label_num)
    else:
        executor.inference = seq_ann_inference


    if set_optimizer:

        executor.optimizer = optimizer_generator(optim_type,model,lr=learning_rate,**kwargs)
        if reduce_lr_on_plateau:
            executor.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(executor.optimizer,verbose=True,
                                                                         threshold=0.1)
    
    executor.loss = None
    
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
    
        executor.loss = label_loss
        
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
