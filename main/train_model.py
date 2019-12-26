import os
import sys
import json
import deepdish as dd
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config "
                        "build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data")
    parser.add_argument("-x","--test_data_path",help="Path of testing data")
    parser.add_argument("-g","--gpu_id",type=str,default='0',help="GPU to used")
    parser.add_argument("--augmentation_max",type=int,default=0)
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--period",default=5,type=int)
    parser.add_argument("--merge",action="store_true")
    parser.add_argument("--model_weights_path")
    parser.add_argument("--executor_weights_path")
    parser.add_argument("--only_train",action='store_true')
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--patient",help="Dafault value is 5. If lower(value) "
                        "is 'none', then model won't be stopped",
                        type=lambda x: int(x) if x.lower() != 'none' else None,default=5)
    parser.add_argument("--frozen_names",type=lambda x:x.split(','),default=None)
    parser.add_argument("--monitor_target")
    parser.add_argument("--map_order_config_path")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import write_fasta, create_folder
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.inference import build_converter
from sequence_annotation.process.utils import param_num
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from main.utils import load_data, get_model, get_executor, GENE_MAP, BASIC_COLOR_SETTING,ANN_TYPES, copy_path
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

def train(saved_root,epoch,model,executor,train_data,val_data=None,
          epoch_start=None,batch_size=None,augmentation_max=None,patient=None,
          color_settings=None,add_seq_fig=True,
          monitor_target=None,period=None,other_callbacks=None):
    with open(os.path.join(saved_root,'param_num.txt'),"w") as fp:
        fp.write("Required-gradient parameters number:{}".format(param_num(model)))
    engine = SeqAnnEngine(ANN_TYPES)
    engine.set_root(saved_root,with_test=False,with_train=True,
                    with_val=val_data is not None)
    other_callbacks = Callbacks()
    if val_data is not None and add_seq_fig:
        seq,ann_seq = _get_max_target_seqs(val_data[0],val_data[1])
        color_settings = color_settings or BASIC_COLOR_SETTING
        seq_fig = engine.get_seq_fig(seq,ann_seq,color_settings=color_settings)
        other_callbacks.add(seq_fig)

    worker = engine.train(model,executor,train_data,val_data=val_data,
                          batch_size=batch_size,epoch=epoch,
                          augmentation_max=augmentation_max,
                          other_callbacks=other_callbacks,
                          checkpoint_kwargs={'monitor_target':monitor_target,
                                             'patient':patient,'period':period})
    return worker

def main(saved_root,model_config_path,executor_config_path,
         train_data_path,val_data_path=None,batch_size=None,
         model_weights_path=None,executor_weights_path=None,
         frozen_names=None,save_distribution=False,merge=False,
         map_order_config_path=None,only_train=False,
         test_data_path=None,epoch=None,**kwargs):

    map_order_config = {}
    if map_order_config_path is not None:
        copy_path(saved_root,map_order_config_path)
        with open(map_order_config_path,"r") as fp:
            map_order_config = json.load(fp)
    
    #Load, parse and save data
    train_data = dd.io.load(train_data_path)
    write_fasta(os.path.join(saved_root,'train.fasta'),train_data[0])
    val_data = None
    test_data = None
    if val_data_path is not None:
        val_data = dd.io.load(val_data_path)
        write_fasta(os.path.join(saved_root,'val.fasta'),val_data[0])
        
    if test_data_path is not None:
        test_data = dd.io.load(test_data_path)
        write_fasta(os.path.join(saved_root,'test.fasta'),test_data[0])
        
    data = train_data, val_data, test_data
    data_path = os.path.join(saved_root,"data.h5")
    
    dd.io.save(data_path,data)

    #Create model
    model = get_model(model_config_path,model_weights_path,frozen_names)
    model.save_distribution = save_distribution

    #Create executor
    with open(executor_config_path,"r") as fp:
        executor_config = json.load(fp)
    
    executor = get_executor(model,executor_weights_path=executor_weights_path,
                            **executor_config)
    
    ann_vec2info_converter = build_converter(ANN_TYPES,GENE_MAP)

    try:
        train(saved_root,epoch,model,executor,train_data,val_data,
              batch_size=batch_size,**map_order_config,**kwargs)     
    except RuntimeError:
        raise Exception("Something wrong ocuurs in {}".format(saved_root))
        
    #Test
    if not only_train:
        executor = get_executor(model,set_loss=False,set_optimizer=False,**executor_config)
        test_paths = ['test_on_train']
        data_list = [train_data]
        if val_data_path is not None:
            test_paths.append('test_on_val')
            data_list.append(val_data)
        if test_data_path is not None:
            test_paths.append('test_on_test')
            data_list.append(test_data)

        for path,data in zip(test_paths,data_list):
            path = os.path.join(saved_root,path)
            create_folder(path)
            test(path,model,executor,data,
                 batch_size=batch_size,use_gffcompare=True,
                 ann_vec2info_converter=ann_vec2info_converter,
                 merge=merge,**map_order_config)
    
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