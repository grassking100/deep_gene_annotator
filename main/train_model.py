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
    parser.add_argument("-g","--gpu_id",type=str,default='0',help="GPU to used")
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
    parser.add_argument("--map_order_config_path")
    parser.add_argument("--use_gffcompare",action="store_true")
    parser.add_argument("--monitor_target",default='val_loss')
    parser.add_argument("--period",default=5,type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_fasta, create_folder
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.callback import ModelCheckpoint, ExecutorCheckpoint,ModelExecutorCheckpoint
from sequence_annotation.process.inference import AnnVec2InfoConverter
from sequence_annotation.process.utils import param_num
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from main.utils import load_data, get_model, get_executor, GENE_MAP, BASIC_COLOR_SETTING,ANN_TYPES, SIMPLIFY_MAP, copy_path
from main.test_model import test

def _get_max_target_seqs(seqs,ann_seqs,seq_fig_target=None):
    max_count = 0
    selected_id = None
    for ann_seq in ann_seqs:
        count = class_count(ann_seq)[seq_fig_target or 'intron']
        if max_count <= count:
            max_count = count
            selected_id = ann_seq.id
    return seqs[selected_id],ann_seqs[selected_id]

def train(model,executor,train_data,val_data=None,saved_root=None,
          epoch=None,batch_size=None,augmentation_max=None,patient=None,
          color_settings=None,seq_fig_target=None,add_seq_fig=True,
          add_grad=True,monitor_target=None,period=None,
          use_gffcompare=False,ann_vec2info_converter=None,
          other_callbacks=None):
    
    epoch_start = 0
    monitor_target = monitor_target or 'val_loss'
    period = period or 5
    
    if saved_root is not None:
        last_status_path = os.path.join(saved_root,'last_model.status')
        latest_status_path = os.path.join(saved_root,'latest_model.status')
        if os.path.exists(latest_status_path):
            with open(latest_status_path,"r") as fp:
                latest_status = json.load(fp)
                epoch_start = max(epoch_start,latest_status['epoch'])

        if os.path.exists(last_status_path):
            with open(last_status_path,"r") as fp:
                last_status = json.load(fp)
                epoch_start = max(epoch_start,last_status['epoch'])

        with open(os.path.join(saved_root,'param_num.txt'),"w") as fp:
            fp.write("Required-gradient parameters number:{}".format(param_num(model)))

    engine = SeqAnnEngine(ann_types=ANN_TYPES)
    engine.ann_vec2info_converter = ann_vec2info_converter
    engine.use_gffcompare = use_gffcompare

    if saved_root is not None:
        engine.set_root(saved_root,with_test=False,with_val=val_data is not None)
    engine.executor = executor
    engine.train_seqs,engine.train_ann_seqs = train_data
    other_callbacks = other_callbacks or []
    if val_data is not None:
        val_seqs,val_ann_seqs = val_data
        engine.val_seqs,engine.val_ann_seqs = val_seqs,val_ann_seqs
        if saved_root is not None:
            if add_seq_fig:
                seq,ann_seq = _get_max_target_seqs(val_seqs,val_ann_seqs,seq_fig_target=seq_fig_target)
                engine.add_seq_fig(seq,ann_seq,
                                   color_settings=color_settings or BASIC_COLOR_SETTING)

    if saved_root is not None:
        model_checkpoint = ModelCheckpoint(target=monitor_target,optimize_min=True,
                                           patient=patient,save_best_weights=True,
                                           restore_best_weights=True,path=saved_root,
                                           period = period)
        executor_checkpoint = ExecutorCheckpoint(path=saved_root,period = period)
        checkpoint = ModelExecutorCheckpoint(model_checkpoint,executor_checkpoint)
        other_callbacks.append(checkpoint)
    
    engine.other_callbacks.add(other_callbacks)

    worker = engine.train(model,batch_size=batch_size,epoch=epoch,
                          augmentation_max=augmentation_max,
                          add_grad=add_grad,
                          epoch_start=epoch_start)
    return worker

def main(saved_root,model_config_path,executor_config_path,
         train_data_path,val_data_path=None,batch_size=None,
         model_weights_path=None,frozen_names=None,save_distribution=False,
         use_gffcompare=False,test_wtih_fix_boundary=False,
         map_order_config_path=None,only_train=False,**kwargs):

    map_order_config = {}
    if map_order_config_path is not None:
        copy_path(saved_root,map_order_config_path)
        with open(map_order_config_path,"r") as fp:
            map_order_config = json.load(fp)
    
    #Load, parse and save data
    train_data = dd.io.load(train_data_path)
    train_data = train_data[0],train_data[1]
    write_fasta(os.path.join(saved_root,'train.fasta'),train_data[0])
    val_data = None
    if val_data_path is not None:
        val_data = dd.io.load(val_data_path)
        val_data = val_data[0],val_data[1]
        write_fasta(os.path.join(saved_root,'val.fasta'),val_data[0])
        
    data = train_data, val_data
    data_path = os.path.join(saved_root,"data.h5")
    
    if not os.path.exists(data_path):
        dd.io.save(data_path,data)

    #Create model
    model = get_model(model_config_path,model_weights_path,frozen_names)
    model.save_distribution = save_distribution

    #Create executor
    with open(executor_config_path,"r") as fp:
        executor_config = json.load(fp)
    
    executor = get_executor(model,**executor_config)
    
    ann_vec2info_converter = AnnVec2InfoConverter(ANN_TYPES,GENE_MAP)
    
    try:
        train(model,executor,train_data,val_data,
              batch_size=batch_size,saved_root=saved_root,
              use_gffcompare=use_gffcompare,
              ann_vec2info_converter=ann_vec2info_converter,
              **map_order_config,**kwargs)
              
    except RuntimeError:
        raise Exception("Something wrong ocuurs in {}".format(saved_root))
        
    #Test
    if not only_train:
        executor = get_executor(model,set_loss=False,set_optimizer=False,**executor_config)
        test_on_train_path = os.path.join(saved_root,'test_on_train')
        create_folder(test_on_train_path)
        test(model,executor,train_data,
             batch_size=batch_size,saved_root=test_on_train_path,
             ann_vec2info_converter=ann_vec2info_converter,
             use_gffcompare=True,fix_boundary=test_wtih_fix_boundary,
             **map_order_config)
        if val_data_path is not None:
            test_on_val_path = os.path.join(saved_root,'test_on_val')
            create_folder(test_on_val_path)
            test(model,executor,val_data,
                 batch_size=batch_size,saved_root=test_on_val_path,
                 ann_vec2info_converter=ann_vec2info_converter,
                 use_gffcompare=True,fix_boundary=test_wtih_fix_boundary,
                 **map_order_config)
    
if __name__ == '__main__':
    #Create folder
    create_folder(args.saved_root)
    #Save setting
    setting_path = os.path.join(args.saved_root,"main_setting.json")
    with open(setting_path,"w") as fp:
        setting = vars(args)
        json.dump(setting, fp, indent=4)

    copy_path(args.saved_root,sys.argv[0])
    copy_path(args.saved_root,args.model_config_path)
    copy_path(args.saved_root,args.executor_config_path)

    kwargs = dict(setting)
    del kwargs['saved_root']
    del kwargs['model_config_path']
    del kwargs['executor_config_path']
    del kwargs['gpu_id']
    main(args.saved_root,args.model_config_path,args.executor_config_path,**kwargs)