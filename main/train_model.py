import os
import sys
from argparse import ArgumentParser
import deepdish as dd
import json

if __name__ != '__main__':
    import torch
    torch.backends.cudnn.benchmark = True
    sys.path.append("/home/sequence_annotation")
    from sequence_annotation.utils.fasta import write_fasta
    from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
    from sequence_annotation.pytorch.callback import EarlyStop
    from sequence_annotation.pytorch.executor import BasicExecutor
    from sequence_annotation.pytorch.model import seq_ann_inference
    from main.utils import load_data, get_model, get_executor, GENE_MAP, BASIC_COLOR_SETTING
    from main.test_model import test

def train(model,executor,train_data,val_data=None,
          saved_root=None,epoch=None,batch_size=None,augmentation_max=None):
    facade = SeqAnnFacade()
    facade.use_gffcompare = False
    facade.alt = False
    if saved_root is not None:
        facade.set_root(saved_root,with_test=False)
    facade.executor = executor
    facade.simplify_map = GENE_MAP
    facade.train_seqs,facade.train_ann_seqs = train_data
    if val_data is not None:
        val_seqs,val_ann_seqs = val_data
        facade.val_seqs,facade.val_ann_seqs = val_seqs,val_ann_seqs

    if saved_root is not None and val_data is not None:
        max_len = None
        selected_id = None
        for seq in val_ann_seqs:
            if max_len is None:
                max_len = len(seq)
                selected_id = seq.id
            elif max_len < len(seq):
                max_len = len(seq)
                selected_id = seq.id
        facade.add_seq_fig(val_seqs[selected_id],val_ann_seqs[selected_id],
                           color_settings=BASIC_COLOR_SETTING)

    ealry_stop = EarlyStop(target='val_loss',optimize_min=True,patient=5,
                           save_best_weights=True,restore_best_weights=True,
                           path=saved_root)
    facade.other_callbacks.add(ealry_stop)    
    record = facade.train(model,batch_size=batch_size,epoch=epoch,augmentation_max=augmentation_max)
    return record

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config",help="Model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_id_path",help="Path of training id data",required=True)
    parser.add_argument("-v","--val_id_path",help="Path of validation id data")
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used")
    parser.add_argument("--use_naive",action="store_true")
    parser.add_argument("--max_len",type=int,default=-1,help="Sequences' max length, if it is negative then it will be ignored")
    parser.add_argument("--min_len",type=int,default=0,help="Sequences' min length")
    parser.add_argument("--augmentation_max",type=int,default=0)
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--learning_rate",type=float,default=1e-3)
    parser.add_argument("--intron_coef",type=float,default=1)
    parser.add_argument("--other_coef",type=float,default=1)
    parser.add_argument("--nontranscript_coef",type=float,default=0)
    parser.add_argument("--transcript_output_mask",action="store_true")
    parser.add_argument("--transcript_answer_mask",action="store_true")
    parser.add_argument("--mean_by_mask",action="store_true")
    parser.add_argument("--model_weights_path")
    parser.add_argument("--disrim_learning_rate",type=float,default=1e-3)
    parser.add_argument("--gamma",type=int,default=0)
    parser.add_argument("--only_train",action='store_true')
    args = parser.parse_args()
    script_path = sys.argv[0]
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id
    ###Load library
    import torch
    torch.backends.cudnn.benchmark = True
    sys.path.append("/home/sequence_annotation")
    from sequence_annotation.utils.fasta import write_fasta
    from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
    from sequence_annotation.pytorch.callback import EarlyStop
    from sequence_annotation.pytorch.executor import BasicExecutor
    from sequence_annotation.pytorch.model import seq_ann_inference
    from main.utils import load_data, get_model, get_executor, GENE_MAP, BASIC_COLOR_SETTING
    from main.test_model import test
    ###
    
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
    
    max_len = None if args.max_len < 0 else args.max_len

    print("Load and parse data")
    data_path = os.path.join(saved_root,"data.h5")
    if os.path.exists(data_path):    
        train_data, val_data = dd.io.load(data_path)
    else:
        data = load_data(args.fasta_path,args.ann_seqs_path,
                         args.train_id_path,args.val_id_path,
                         args.min_len,max_len)
        dd.io.save(data_path,data)
        train_data, val_data = data
        
    model = get_model(args.model_config,args.model_weights_path)
    model.save_distribution = False

    executor = get_executor(model,**vars(args))
    write_fasta(os.path.join(saved_root,'train.fasta'),train_data[0])
    if val_data is not None:
        write_fasta(os.path.join(saved_root,'val.fasta'),val_data[0])

    train(model,executor,train_data,val_data,saved_root,
          args.epoch,args.batch_size,args.augmentation_max)
    
    if not args.only_train:
        executor = BasicExecutor()
        executor.loss = None
        if not args.use_naive:
            executor.inference = seq_ann_inference
        test_on_train_path = os.path.join(args.saved_root,'test_on_train')
        test_on_val_path = os.path.join(args.saved_root,'test_on_val')
        
        if not os.path.exists(test_on_train_path):
            os.mkdir(test_on_train_path)
        
        if not os.path.exists(test_on_val_path):
            os.mkdir(test_on_val_path)

        test(model,executor,train_data,args.batch_size,test_on_train_path)
        test(model,executor,val_data,args.batch_size,test_on_val_path)
    