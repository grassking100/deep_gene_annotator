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

def test(model,data,saved_root,executor,batch_size):
    facade = SeqAnnFacade()
    facade.use_gffcompare = True
    facade.alt = False
    facade.set_root(saved_root,with_train=False,with_val=False)
    facade.executor = executor
    facade.simplify_map = gene_map
    facade.test_seqs,facade.test_ann_seqs = data
    train_record = facade.test(model,batch_size=batch_size)

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config",help="Model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-i","--id_path",help="Path of id data",required=True)
    parser.add_argument("-w","--model_weights_path",required=True)
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used")
    parser.add_argument("--max_len",type=int,default=10000,help="Sequences' max length, if it is negative then it will be ignored")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--use_naive",action="store_true")
    
    args = parser.parse_args()
    max_len = args.max_len
    if max_len < 0:
        max_len = None
    script_path = sys.argv[0]
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id
    saved_root = args.saved_root
    setting = vars(args)
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)

    setting_path = os.path.join(saved_root,"test_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(setting, fp, indent=4)
    
    #Load library
    import torch
    torch.backends.cudnn.benchmark = True
    from torch import nn
    sys.path.append("/home/sequence_annotation")
    from sequence_annotation.genome_handler.load_data import load_data
    from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
    from sequence_annotation.pytorch.executer import BasicExecutor
    from sequence_annotation.pytorch.model import SeqAnnBuilder
    from sequence_annotation.pytorch.model import seq_ann_inference

    builder = SeqAnnBuilder()
    with open(args.model_config,"r") as fp:
        builder.config = json.load(fp)

    model = builder.build().cuda()    
    weight = torch.load(args.model_weights_path)
    model.load_state_dict(weight)
        
    executor = BasicExecutor()
    executor.loss = None
    if not args.use_naive:
        executor.inference = seq_ann_inference

    ids = list(pd.read_csv(args.id_path,header=None)[0])
    data = load_data(args.fasta_path,args.ann_seqs_path,[ids],
                     max_len=max_len,simplify_map=simplify_map,
                     before_mix_simplify_map=before_mix_simplify_map)
    
    test(model,data[0],saved_root,executor,args.batch_size)
