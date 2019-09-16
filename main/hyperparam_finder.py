import os
import sys
sys.path.append("./sequence_annotation")
import json
import pickle
import pandas as pd
import deepdish as dd
from argparse import ArgumentParser
from hyperopt import fmin, tpe, hp, Trials
import torch
from torch import nn
torch.backends.cudnn.benchmark = True
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.space_evaluate import SpaceEvaluator
from sequence_annotation.utils.utils import write_fasta
from sequence_annotation.genome_handler.load_data import load_data
from sequence_annotation.genome_handler.alt_count import max_alt_count
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer

def train(train_data,val_data,saved_root,batch_size,epoch,max_evals,parameter_sharing):
    engine = SeqAnnEngine()
    engine.is_verbose_visible = True
    engine.use_gffcompare = False
    train_data,val_data = data
    engine.train_seqs,engine.train_ann_seqs = train_data
    engine.val_seqs,engine.val_ann_seqs = val_data
    write_fasta(os.path.join(saved_root,'train.fasta'),engine.train_seqs)
    write_fasta(os.path.join(saved_root,'val.fasta'),engine.val_seqs)
    space = {
        'feature_block':{
            'num_layers':4,
            'cnns_setting':{'out_channels':hp.quniform('feature_channel_size', 32, 64,1),
                            'kernel_size':hp.quniform('feature_kernel_size', 32,256, 1)}
        },
        'relation_block':{
             'rnns_setting':{
                 'num_layers':4,
                 'hidden_size':32,
                 'bidirectional':True
             },
             'rnns_type':nn.GRU
        },
        'projection_layer':{
             'kernel_size':hp.quniform('projection_kernel_size', 1, 3, 1)
        },
        'in_channels':4,
        'inference_method':hp.choice('inference_method',
                                     [
                                         {'type':'naive'},
                                         {'type':'hierarchy',
                                          'coef':{
                                              'intron_coef':hp.uniform('intron_coef', 0.1, 10),
                                              'other_coef':hp.uniform('other_coef', 0.1, 10)
                                          }
                                         }
                                     ])
         
    }
    trials = Trials()
    space_evaluator = SpaceEvaluator(engine,saved_root)
    space_evaluator.parameter_sharing = parameter_sharing
    space_evaluator.target_min = False
    space_evaluator.eval_target = 'val_macro_F1'
    space_evaluator.batch_size = batch_size
    space_evaluator.epoch = epoch
    space_evaluator.patient = 16
    best = fmin(space_evaluator.objective, space, algo=tpe.suggest, max_evals=max_evals,trials=trials)
    best_path = os.path.join(saved_root,"best.json")
    space_result_path = os.path.join(saved_root,"space_result.txt")
    records_path = os.path.join(saved_root,"records.json")
    trials_path = os.path.join(saved_root,"trials.pkl")

    with open(best_path,"w") as fp:
        json.dump(best,fp)
    
    with open(space_result_path,"w") as fp:
        fp.write(str(space_evaluator.space_result))

    with open(records_path,"w") as fp:
        json.dump(space_evaluator.records,fp)
        
    with open(trials_path, 'wb') as fp:
        pickle.dump(trials,fp)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_id_path",help="Path of training id data",required=True)
    parser.add_argument("-v","--val_id_path",help="Path of validation id data",required=True)
    parser.add_argument("-b","--batch_size",help="Batch size",default=32,type=int,required=False)
    parser.add_argument("-e","--epoch",help="Epoch",default=16,type=int,required=False)
    parser.add_argument("-x","--max_evals",help="Eval space number",default=32,type=int,required=False)
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used",required=False)
    parser.add_argument("-p","--parameter_sharing",default='T',required=False,
                        type=lambda x: x == 'T',help="To use parameter sharing or not")
    parser.add_argument("-m","--max_len",type=int,default=10000,
                        help="Seqeunces' max length",required=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not os.path.exists(args.saved_root):
        os.mkdir(saved_root)
    setting = vars(args)
    setting_path = os.path.join(saved_root,"main_setting.json")
    with open(setting_path,"w") as outfile:
        json.dump(setting, outfile)
    command = 'cp -t {} {}'.format(args.saved_root,sys.argv[0])
    os.system(command)
    train_data,val_data = load_data(args.fasta_path,args.ann_seqs_path,args.max_len,
                                    args.train_id_path,args.val_id_path,args.saved_root)
    train(train_data,val_data,args.batch_size,args.epoch,args.max_evals,args.parameter_sharing)