import os
import sys
import json
import deepdish as dd
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data")
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used")
    parser.add_argument("-w","--model_weights_path")
    parser.add_argument("--augmentation_max",type=int,default=0)
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--only_train",action='store_true')
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--test_wtih_fix_boundary",action="store_true")
    parser.add_argument("--patient",help="Dafault value is 5. If lower(value) is 'none', then model won't be stopped",
                        type=lambda x: int(x) if x.lower() != 'none' else None,default=5)
    parser.add_argument("--frozen_names",type=lambda x:x.split(','),default=None)
    #parser.add_argument("--load_previous",action='store_true')
    parser.add_argument("--map_order_config_path")
    parser.add_argument("--use_gffcompare",action="store_true")
    parser.add_argument("--monitor_target",default='val_loss')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_fasta, create_folder
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.callback import ModelCheckpoint, ExecutorCheckpoint
from sequence_annotation.process.inference import seq_ann_inference
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from main.utils import load_data, get_model, get_executor, GENE_MAP, BASIC_COLOR_SETTING,ANN_TYPES, SIMPLIFY_MAP, copy_path
from main.test_model import test

CHECKPOINT_PERIOD = 5

def train(model,executor,train_data,val_data=None,saved_root=None,
          epoch=None,batch_size=None,augmentation_max=None,patient=None,
          gene_map=None,color_settings=None,channel_order=None,seq_fig_target=None,
          ann_types=None,use_gffcompare=False,other_callbacks=None,
          add_grad=True,add_seq_fig=True,epoch_start=None,monitor_target=None):
    channel_order = channel_order or list(train_data[1].ANN_TYPES)
    engine = SeqAnnEngine(ann_types=ann_types or ANN_TYPES,channel_order=channel_order)
    engine.use_gffcompare = use_gffcompare
    monitor_target = monitor_target or 'val_loss'
    if saved_root is not None:
        engine.set_root(saved_root,with_test=False,with_val=val_data is not None)
    engine.executor = executor
    engine.gene_map = gene_map or GENE_MAP
    engine.train_seqs,engine.train_ann_seqs = train_data
    other_callbacks = other_callbacks or []
    if val_data is not None:
        val_seqs,val_ann_seqs = val_data
        engine.val_seqs,engine.val_ann_seqs = val_seqs,val_ann_seqs
        if saved_root is not None:
            if add_seq_fig:
                max_count = 0
                selected_id = None
                for seq in val_ann_seqs:
                    count = class_count(seq)[seq_fig_target or 'intron']
                    if max_count <= count:
                        max_count = count
                        selected_id = seq.id
                engine.add_seq_fig(val_seqs[selected_id],val_ann_seqs[selected_id],
                                   color_settings=color_settings or BASIC_COLOR_SETTING)

    if saved_root is not None:
        model_checkpoint = ModelCheckpoint(target=monitor_target,optimize_min=True,
                                           patient=patient,save_best_weights=True,
                                           restore_best_weights=True,path=saved_root,
                                           period = CHECKPOINT_PERIOD)
        other_callbacks.append(model_checkpoint)
        checkpoint = ExecutorCheckpoint(path=saved_root,period = CHECKPOINT_PERIOD)
        other_callbacks.append(checkpoint)
    
    engine.other_callbacks.add(other_callbacks)

    worker = engine.train(model,batch_size=batch_size,epoch=epoch,
                          augmentation_max=augmentation_max,
                          add_grad=add_grad,
                          epoch_start=epoch_start)
    return worker

if __name__ == '__main__':
    #Create folder
    if not os.path.exists(args.saved_root):
        os.system("mkdir -p {}".format(args.saved_root))
    
    model_weights_path = args.model_weights_path
    epoch_start = 0
    
    #Save setting
    setting_path = os.path.join(args.saved_root,"main_setting.json")
    with open(setting_path,"w") as fp:
        setting = vars(args)
        json.dump(setting, fp, indent=4)

    copy_path(args.saved_root,sys.argv[0])
    copy_path(args.saved_root,args.model_config_path)
    copy_path(args.saved_root,args.executor_config_path)
    
    #Create map_order_config
    map_order_config = {}
    map_order_config['data_simplify_map'] = SIMPLIFY_MAP
    if args.map_order_config_path is not None:
        copy_path(args.saved_root,args.map_order_config_path)
        with open(args.map_order_config_path,"r") as fp:
            map_order_config = json.load(fp)
    
    data_simplify_map = map_order_config['data_simplify_map']
    del map_order_config['data_simplify_map']
    
    #Load, parse and save data
    train_data = dd.io.load(args.train_data_path)
    train_data = train_data[0],simplify_genome(train_data[1],data_simplify_map)
    write_fasta(os.path.join(args.saved_root,'train.fasta'),train_data[0])
    val_data = None
    if args.val_data_path is not None:
        val_data = dd.io.load(args.val_data_path)
        val_data = val_data[0],simplify_genome(val_data[1],data_simplify_map)
        write_fasta(os.path.join(args.saved_root,'val.fasta'),val_data[0])
        
    data = train_data, val_data
    data_path = os.path.join(args.saved_root,"data.h5")
    
    if not os.path.exists(data_path):
        dd.io.save(data_path,data)

    last_status_path = os.path.join(args.saved_root,'last_model.status')
    latest_status_path = os.path.join(args.saved_root,'latest_model.status')
    if os.path.exists(latest_status_path):
        #Load latest model
        with open(latest_status_path,"r") as fp:
            latest_status = json.load(fp)
        if os.path.exists(last_status_path):
            with open(last_status_path,"r") as fp:
                last_status = json.load(fp)
            if last_status['epoch'] <= latest_status['epoch']:
                epoch_start = latest_status['epoch']
            else:
                epoch_start = last_status['epoch']
        else:
            epoch_start = latest_status['epoch']

    #Create model
    model = get_model(args.model_config_path,model_weights_path,args.frozen_names)
    model.save_distribution = args.save_distribution

    #Create executor
    with open(args.executor_config_path,"r") as fp:
        executor_config = json.load(fp)
    
    executor = get_executor(model,**executor_config)
    #executor_status_path = os.path.join(args.saved_root,'last_executor.pth')
    #if os.path.exists(executor_status_path):
    #    executor.load_state_dict(torch.load(executor_status_path))

    #Train
    #best_model_path = os.path.join(args.saved_root,'best_model.pth')
    #last_model_path = os.path.join(args.saved_root,'last_model.pth')
    #if not os.path.exists(best_model_path) and not os.path.exists(last_model_path):
    train(model,executor,train_data,val_data,args.saved_root,
          args.epoch,args.batch_size,args.augmentation_max,
          patient=args.patient,use_gffcompare=args.use_gffcompare,
          epoch_start=epoch_start,
          monitor_target=args.monitor_target,
          **map_order_config)
    #elif os.path.exists(best_model_path):
    #    model.load_state_dict(torch.load(best_model_path))
    #else:
    #    model.load_state_dict(torch.load(last_model_path))
    
    #Test
    if not args.only_train:
        executor = get_executor(model,set_loss=False,set_optimizer=False,**executor_config)
        test_on_train_path = os.path.join(args.saved_root,'test_on_train')
        create_folder(test_on_train_path)
        test(model,executor,train_data,args.batch_size,test_on_train_path,**map_order_config)
        if args.val_data_path is not None:
            test_on_val_path = os.path.join(args.saved_root,'test_on_val')
            create_folder(test_on_val_path)
            test(model,executor,val_data,args.batch_size,test_on_val_path,**map_order_config)
        
        if args.test_wtih_fix_boundary:
            test_on_train_by_fixed_path = os.path.join(args.saved_root,'test_on_train_by_fixed')
            create_folder(test_on_train_by_fixed_path)
            test(model,executor,train_data,args.batch_size,test_on_train_by_fixed_path,
                 fix_boundary=True,**map_order_config)
            if args.val_data_path is not None:
                test_on_val_by_fixed_path = os.path.join(args.saved_root,'test_on_val_by_fixed')
                create_folder(test_on_val_by_fixed_path)
                test(model,executor,val_data,args.batch_size,test_on_val_by_fixed_path,
                     fix_boundary=True,**map_order_config)
