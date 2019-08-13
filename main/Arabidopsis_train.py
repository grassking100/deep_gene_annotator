import os
import sys
from argparse import ArgumentParser
import deepdish as dd
import pandas as pd
import json

before_mix_simplify_map = {'exon':['exon'],'intron':['intron','alt_accept','alt_donor'],'other':['other']}
simplify_map = {'exon':['exon'],'intron':['intron'],'other':['other']}
gene_map = {'gene':['exon','intron'],'other':['other']}
color_settings={'other':'blue','exon':'red','intron':'yellow'}

def _load_data(fasta_path,ann_seqs_path,max_len,train_id_path,val_id_path,saved_root):
    print("Load and parse data")
    data_path = os.path.join(saved_root,"data.h5")
    if os.path.exists(data_path):    
        data = dd.io.load(data_path)
    else:
        train_ids = list(pd.read_csv(train_id_path,header=None)[0])
        val_ids = None
        if val_id_path is not None:
            val_ids = list(pd.read_csv(val_id_path,header=None)[0])
        data = load_data(fasta_path,ann_seqs_path,train_ids,val_ids,max_len=max_len,
                         simplify_map=simplify_map,before_mix_simplify_map=before_mix_simplify_map)
        dd.io.save(data_path,data)
    train_data,val_data,_ = data
    return train_data,val_data

def train(model,train_data,val_data,saved_root,executor,epoch,batch_size,augmentation_max):
    facade = SeqAnnFacade()
    facade.use_gffcompare = False
    facade.alt = False
    facade.set_root(saved_root,with_test=False)
    facade.executor = executor
    facade.simplify_map = gene_map
    facade.train_seqs,facade.train_ann_seqs = train_data
    write_fasta(os.path.join(saved_root,'train.fasta'),facade.train_seqs)
    
    if val_data is not None:
        facade.val_seqs,facade.val_ann_seqs = val_data[:2]
        facade.add_seq_fig(*val_data[2:],color_settings=color_settings)
        ealry_stop = EarlyStop(target='val_loss',optimize_min=True,patient=5,
                               save_best_weights=True,restore_best_weights=True,
                               path=saved_root)
        facade.other_callbacks.add(ealry_stop)
        write_fasta(os.path.join(saved_root,'val.fasta'),facade.val_seqs)
    
    train_record = facade.train(model,batch_size=batch_size,epoch=epoch,augmentation_max=augmentation_max)

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config",help="Model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_id_path",help="Path of training id data",required=True)
    parser.add_argument("-v","--val_id_path",help="Path of validation id data",default=None,required=False)
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used",required=False)
    parser.add_argument("--use_naive",action="store_true", required=False)
    parser.add_argument("--max_len",type=int,default=10000,help="Seqeunces' max length",required=False)
    parser.add_argument("--augmentation_max",type=int,default=0,required=False)
    parser.add_argument("--epoch",type=int,default=100,required=False)
    parser.add_argument("--batch_size",type=int,default=32,required=False)
    parser.add_argument("--learning_rate",type=float,default=1e-3,required=False)
    parser.add_argument("--intron_coef",type=float,default=1,required=False)
    parser.add_argument("--other_coef",type=float,default=1,required=False)
    parser.add_argument("--nontranscript_coef",type=float,default=0,required=False)
    parser.add_argument("--transcript_output_mask",action="store_true", required=False)
    parser.add_argument("--transcript_answer_mask",action="store_true", required=False)
    parser.add_argument("--mean_by_mask",action="store_true", required=False)
    parser.add_argument("--model_weights_path", type=str, required=False)
    parser.add_argument("--disrim_learning_rate",type=float,default=1e-3,required=False)
    args = parser.parse_args()
    script_path = sys.argv[0]
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id
    saved_root = args.saved_root
    setting = vars(args)
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)

    setting_path = os.path.join(saved_root,"main_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(setting, fp, indent=4)

    command = 'cp -t {} {}'.format(saved_root,script_path)
    os.system(command)
    
    command = 'cp -t {} {}'.format(saved_root,args.model_config)
    os.system(command)
    
    #Load library
    import torch
    torch.backends.cudnn.benchmark = True
    from torch import nn
    sys.path.append("../sequence_annotation")
    from sequence_annotation.utils.fasta import write_fasta
    from sequence_annotation.genome_handler.load_data import load_data
    from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
    from sequence_annotation.pytorch.loss import SeqAnnLoss
    from sequence_annotation.pytorch.executer import BasicExecutor, GANExecutor
    from sequence_annotation.pytorch.model import seq_ann_inference, SeqAnnBuilder
    from sequence_annotation.pytorch.model import seq_ann_reverse_inference
    from sequence_annotation.pytorch.callback import EarlyStop

    builder = SeqAnnBuilder()
    with open(args.model_config,"r") as fp:
        builder.config = json.load(fp)
    model = builder.build().cuda()
    
    if args.model_weights_path is not None:
        weight = torch.load(args.model_weights_path)
        model.load_state_dict(weight)
        
    if builder.use_discrim:
        executor = GANExecutor()
        executor.reverse_inference = seq_ann_reverse_inference
        executor.optimizer = (torch.optim.Adam(model.gan.parameters(),
                                               lr=args.learning_rate),
                              torch.optim.Adam(model.discrim.parameters(),
                                               lr=args.disrim_learning_rate))
    else:
        executor = BasicExecutor()
        executor.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=args.learning_rate)

    if not args.use_naive:  
        executor.loss = SeqAnnLoss(intron_coef=args.intron_coef,other_coef=args.other_coef,
                                   nontranscript_coef=args.nontranscript_coef,
                                   transcript_answer_mask=args.transcript_answer_mask,
                                   transcript_output_mask=args.transcript_output_mask,
                                   mean_by_mask=args.mean_by_mask)
        executor.inference = seq_ann_inference
        
    train_data, val_data = _load_data(args.fasta_path,args.ann_seqs_path,args.max_len,
                                      args.train_id_path,args.val_id_path,args.saved_root)
    train(model,train_data,val_data,saved_root,executor,args.epoch,
          args.batch_size,args.augmentation_max)
