import os
import sys
from argparse import ArgumentParser
import json

if __name__ != '__main__':
    import torch
    torch.backends.cudnn.benchmark = True
    sys.path.append("/home/sequence_annotation")
    from sequence_annotation.genome_handler.load_data import load_data
    from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
    from sequence_annotation.pytorch.executor import BasicExecutor
    from sequence_annotation.pytorch.model import seq_ann_inference
    from main.utils import load_data,get_model, GENE_MAP
    
def test(model,executor,data,batch_size=None,saved_root=None):
    facade = SeqAnnFacade()
    facade.use_gffcompare = True
    facade.alt = False
    if saved_root is not None:
        facade.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)
    facade.executor = executor
    facade.simplify_map = GENE_MAP
    facade.test_seqs,facade.test_ann_seqs = data
    record = facade.test(model,batch_size=batch_size)
    return record

def main(id_path,fasta_path,ann_seqs_path,model_config_path,model_weights_path,
         use_naive=True,batch_size=None,max_len=None,saved_root=None):
    model = get_model(model_config_path,model_weights_path=model_weights_path)
    executor = BasicExecutor()
    executor.loss = None
    if not use_naive:
        executor.inference = seq_ann_inference
    data = load_data(fasta_path,ann_seqs_path,id_path,max_len=max_len)
    record = test(model,executor,data[0],batch_size,saved_root)
    return record

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config",help="Model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-i","--id_path",help="Path of id data",required=True)
    parser.add_argument("-w","--model_weights_path",required=True)
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used")
    parser.add_argument("--max_len",type=int,default=-1,help="Sequences' max length, if it is negative then it will be ignored")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--use_naive",action="store_true")
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id
    ###Load library
    import torch
    torch.backends.cudnn.benchmark = True
    sys.path.append("/home/sequence_annotation")
    from sequence_annotation.genome_handler.load_data import load_data
    from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
    from sequence_annotation.pytorch.executor import BasicExecutor
    from sequence_annotation.pytorch.model import seq_ann_inference
    from main.utils import load_data,get_model, GENE_MAP
    ###
    setting = vars(args)
    if not os.path.exists(args.saved_root):
        os.mkdir(args.saved_root)

    setting_path = os.path.join(args.saved_root,"test_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(setting, fp, indent=4)

    max_len = None if args.max_len < 0 else args.max_len
        
    main(args.id_path,args.fasta_path,args.ann_seqs_path,
         args.model_config,args.model_weights_path,
         args.use_naive,args.batch_size,max_len,args.saved_root)
