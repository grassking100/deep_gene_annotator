import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data")
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used")
    parser.add_argument("--augmentation_max",type=int,default=0)
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--model_weights_path")
    parser.add_argument("--only_train",action='store_true')
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--test_wtih_fix_boundary",action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import sys
import json
import deepdish as dd
import torch
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_fasta
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.callback import EarlyStop
from sequence_annotation.process.inference import seq_ann_inference
from main.utils import load_data, get_model, get_executor, GENE_MAP, BASIC_COLOR_SETTING
from main.test_model import test

def train(model,executor,train_data,val_data=None,
          saved_root=None,epoch=None,batch_size=None,augmentation_max=None):
    engine = SeqAnnEngine()
    engine.use_gffcompare = False
    if saved_root is not None:
        engine.set_root(saved_root,with_test=False)
    engine.executor = executor
    engine.simplify_map = GENE_MAP
    engine.train_seqs,engine.train_ann_seqs = train_data
    if val_data is not None:
        val_seqs,val_ann_seqs = val_data
        engine.val_seqs,engine.val_ann_seqs = val_seqs,val_ann_seqs
        if saved_root is not None:
            max_len = None
            selected_id = None
            for seq in val_ann_seqs:
                if max_len is None:
                    max_len = len(seq)
                    selected_id = seq.id
                elif max_len < len(seq):
                    max_len = len(seq)
                    selected_id = seq.id
            engine.add_seq_fig(val_seqs[selected_id],val_ann_seqs[selected_id],
                               color_settings=BASIC_COLOR_SETTING)

    ealry_stop = EarlyStop(target='val_loss',optimize_min=True,patient=5,
                           save_best_weights=True,restore_best_weights=True,
                           path=saved_root)
    engine.other_callbacks.add(ealry_stop)    
    record = engine.train(model,batch_size=batch_size,epoch=epoch,augmentation_max=augmentation_max)
    return record

def _copy_path(root,path):
    command = 'cp -t {} {}'.format(root,path)
    os.system(command)    

if __name__ == '__main__':
    #Create folder
    if not os.path.exists(args.saved_root):
        os.mkdir(args.saved_root)
    
    #Save setting
    setting_path = os.path.join(args.saved_root,"main_setting.json")
    with open(setting_path,"w") as fp:
        setting = vars(args)
        json.dump(setting, fp, indent=4)

    _copy_path(args.saved_root,sys.argv[0])
    _copy_path(args.saved_root,args.model_config_path)
    _copy_path(args.saved_root,args.executor_config_path)
    
    #Load, parse and save data
    train_data = dd.io.load(args.train_data_path)
    write_fasta(os.path.join(args.saved_root,'train.fasta'),train_data[0])
    val_data = None
    if args.val_data_path is not None:
        val_data = dd.io.load(args.val_data_path)
        write_fasta(os.path.join(args.saved_root,'val.fasta'),val_data[0])
    data = train_data, val_data
    data_path = os.path.join(args.saved_root,"data.h5")
    if not os.path.exists(data_path):
        dd.io.save(data_path,data)

    #Create model
    model = get_model(args.model_config_path,args.model_weights_path)
    model.save_distribution = args.save_distribution

    #Create executor
    with open(args.executor_config_path,"r") as fp:
        executor_config = json.load(fp)
    
    executor = get_executor(model,**executor_config)

    #Train
    train(model,executor,train_data,val_data,args.saved_root,
          args.epoch,args.batch_size,args.augmentation_max)
    
    #Test
    if not args.only_train:
        executor = get_executor(model,set_loss=False,set_optimizer=False,**executor_config)
        test_on_train_path = os.path.join(args.saved_root,'test_on_train')
        test_on_val_path = os.path.join(args.saved_root,'test_on_val')
        
        if not os.path.exists(test_on_train_path):
            os.mkdir(test_on_train_path)
        
        if not os.path.exists(test_on_val_path):
            os.mkdir(test_on_val_path)

        test(model,executor,train_data,args.batch_size,test_on_train_path)
        test(model,executor,val_data,args.batch_size,test_on_val_path)
        
        if args.test_wtih_fix_boundary:
            test_on_train_by_fixed_path = os.path.join(args.saved_root,'test_on_train_by_fixed')
            test_on_val_by_fixed_path = os.path.join(args.saved_root,'test_on_val_by_fixed')
            if not os.path.exists(test_on_train_by_fixed_path):
                os.mkdir(test_on_train_by_fixed_path)

            if not os.path.exists(test_on_val_by_fixed_path):
                os.mkdir(test_on_val_by_fixed_path)

            test(model,executor,train_data,args.batch_size,test_on_train_by_fixed_path,fix_boundary=True)
            test(model,executor,val_data,args.batch_size,test_on_val_by_fixed_path,fix_boundary=True)
