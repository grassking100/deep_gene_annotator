import os
import sys
from argparse import ArgumentParser
import pandas as pd
import json
from numpy import median
import torch
sys.path.append("/home/sequence_annotation")
from sequence_annotation.genome_handler.load_data import load_data as _load_data
from sequence_annotation.process.executor import ExecutorBuilder
from sequence_annotation.process.model import SeqAnnBuilder
from sequence_annotation.process.utils import get_name_parameter

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

def get_executor(model,optim_type=None,use_native=True,set_loss=True,set_optimizer=True,
                 label_num=None,predict_label_num=None,answer_label_num=None,output_label_num=None,
                 grad_clip=None,grad_norm=None,
                 learning_rate=None,reduce_lr_on_plateau=False,
                 gamma=None,intron_coef=None,other_coef=None,nontranscript_coef=None,
                 transcript_answer_mask=True,transcript_output_mask=False,mean_by_mask=False,
                 target_weight_decay=None,weight_decay_name=None,
                 **kwargs):

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
        builder.set_loss(gamma=gamma,intron_coef=intron_coef,
                         other_coef=other_coef,
                         nontranscript_coef=nontranscript_coef,
                         transcript_answer_mask=transcript_answer_mask,
                         transcript_output_mask=transcript_output_mask,
                         mean_by_mask=mean_by_mask)
        
    return builder.build()

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
