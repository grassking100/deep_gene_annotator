import os
import sys
sys.path.append("./sequence_annotation")
import json
import pandas as pd
import torch
torch.backends.cudnn.benchmark = True
from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
from sequence_annotation.pytorch.trainer import SeqAnnTrainer
from sequence_annotation.data_handler.fasta import write_fasta
from sequence_annotation.utils.load_data import load_data
from sequence_annotation.genome_handler.alt_count import max_alt_count

def train(fasta_path,ann_seqs_path,max_len,train_ids,val_ids,saved_root):
    setting = locals()
    del setting['model']
    setting_path = os.path.join(saved_root,"main_setting.json")
    with open(setting_path,"w") as outfile:
        json.dump(setting, outfile)
    before_mix_simplify_map={'exon':['utr_5','utr_3','cds'],
                             'intron':['intron'],'other':['other']}
    simplify_map={'exon':['exon'],'intron':['intron'],
                  'other':['other'],'alternative':['exon_intron']}
    data_path = os.path.join(saved_root,"data.h5")
    if os.path.exists(data_path):    
        data = dd.io.load(data_path)
    else:    
        data = load_data(fasta_path,ann_seqs_path,max_len,
                         train_ids,val_ids,simplify_map,
                         before_mix_simplify_map=before_mix_simplify_map)
        dd.io.save(data_path,data)

    facade = SeqAnnFacade()
    facade.use_gffcompare = False
    train_data,val_data = data
    facade.train_seqs,facade.train_ann_seqs = train_data
    facade.val_seqs,facade.val_ann_seqs = val_data[:2]                                       
    write_fasta(os.path.join(saved_root,'train.fasta'),facade.train_seqs)
    write_fasta(os.path.join(saved_root,'val.fasta'),facade.val_seqs)
    space = {
        'feature_block':{
            'num_layers':4,
            'out_channels':hp.quniform('feature_channel_size', 8, 64,1),
            'kernel_size':hp.quniform('feature_kernel_size', 8,256, 1)
        },
        'relation_block':{
             'num_layers':4,
             'hidden_size':hp.quniform('relation_hidden_size', 8, 64,1),
             'bidirectional':hp.choice('bidirectional',[True,False]),
             'rnns_type':nn.GRU
        },
        'projection_layer':{
             'kernel_size':hp.quniform('projection_kernel_size', 1, 2, 1)
        },
         'inference_method':{'inference_method':hp.choice('inference_method',
             [
                 {'type':'naive'},
                 {'type':'hierarchy',
                  'coef':{
                      'intron_coef':hp.uniform('intron_coef', 0, 2),
                      'other_coef':hp.uniform('other_coef', 0, 2)
                  }
                 }
             ]
             )
         }
    }
    trials = Trials()
    trainer = SeqAnnTrainer()
    best = fmin(trainer.objective, space, algo=tpe.suggest, max_evals=100,trials=trials)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_id_path",help="Path of training id data",required=True)
    parser.add_argument("-v","--val_id_path",help="Path of validation id data",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used",required=False)
    parser.add_argument("-m","--max_len",type=int,default=10000,
                        help="Seqeunces' max length",required=False)
    args = parser.parse_args()
    train_ids = pd.read_csv(args.train_id_path,header=None)[0]
    val_ids = pd.read_csv(args.val_id_path,header=None)[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)
    command = 'cp -t {} {}'.format(saved_root,sys.argv[0])
    os.system(command)
    train(args.fasta_path,args.ann_seqs_path,args.max_len,train_ids,val_ids,args.saved_root)