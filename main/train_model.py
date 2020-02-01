import os
import sys
import json
import torch
import numpy as np
import random
import deepdish as dd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.utils.utils import create_folder, write_fasta, write_json, read_json
from sequence_annotation.utils.utils import BASIC_GENE_MAP,BASIC_GENE_ANN_TYPES,get_time_str
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.convert_signal_to_gff import build_ann_vec_gff_converter
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.process.model import SeqAnnBuilder
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from main.utils import BASIC_COLOR_SETTING, copy_path,load_data,get_model_executor
from main.test_model import test

DEFAULT_BATCH_SIZE = 32

def _get_max_target_seqs(seqs,ann_seqs,seq_fig_target=None):
    max_count = 0
    selected_id = None
    for ann_seq in ann_seqs:
        count = class_count(ann_seq)[seq_fig_target or 'intron']
        if max_count <= count:
            max_count = count
            selected_id = ann_seq.id
    return seqs[selected_id],ann_seqs[selected_id]

def _get_first_large_data(data,batch_size):
    seqs,ann_container = data
    lengths = {key:len(seq) for key,seq in seqs.items()}
    sorted_length_keys = sorted(lengths,key=lengths.get,reverse=True)
    part_keys = sorted_length_keys[:batch_size]
    part_seqs = dict(zip(part_keys,[seqs[key] for key in part_keys]))
    part_container = AnnSeqContainer()
    part_container.ANN_TYPES = ann_container.ANN_TYPES
    for key in part_keys:
        part_container.add(ann_container.get(key))
        
    return part_seqs,part_container

def check_max_memory_usgae(saved_root,model,executor,train_data,val_data,batch_size=None):
    batch_size = batch_size or DEFAULT_BATCH_SIZE
    
    train_data = _get_first_large_data(train_data,batch_size)
    val_data = _get_first_large_data(val_data,batch_size)
    
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES,is_verbose_visible=False)
    try:
        engine.train(model,executor,train_data,val_data=val_data,batch_size=batch_size,epoch=1)
        del model
        del executor
        del engine
    except RuntimeError:
        path = os.path.join(saved_root,'error.txt')
        with open(path,"w") as fp:
            fp.write("Memory might be fulled at {}\n".format(get_time_str()))
        raise Exception("Memory is fulled")

def train(saved_root,epoch,model,executor,train_data,val_data,
          batch_size=None,augmentation_max=None,patient=None,
          monitor_target=None,period=None,deterministic=False):
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES,shuffle_train_data=not deterministic)    
    engine.set_root(saved_root,with_test=False,with_train=True,with_val=True)
    other_callbacks = Callbacks()
    seq,ann_seq = _get_max_target_seqs(val_data[0],val_data[1])
    seq_fig = engine.get_seq_fig(seq,ann_seq,color_settings=BASIC_COLOR_SETTING)
    other_callbacks.add(seq_fig)
    checkpoint_kwargs={'monitor_target':monitor_target,'patient':patient,'period':period}
    engine.train(model,executor,train_data,val_data=val_data,
                 batch_size=batch_size,epoch=epoch,
                 augmentation_max=augmentation_max,
                 other_callbacks=other_callbacks,
                 checkpoint_kwargs=checkpoint_kwargs)

def main(saved_root,model_config_path,executor_config_path,
         train_data_path,val_data_path,
         test_data_path=None,batch_size=None,epoch=None,
         frozen_names=None,save_distribution=False,only_train=False,
         train_answer_path=None,#train_region_path=None,
         val_answer_path=None,#val_region_path=None,
         test_answer_path=None,#test_region_path=None,
         region_table_path=None,deterministic=False,**kwargs):
    
    copied_paths = [model_config_path,executor_config_path,train_data_path,val_data_path]
    if args.test_data_path is not None:
        copied_paths.append(test_data_path)
    for path in copied_paths:
        copy_path(saved_root,path)
    
    if deterministic:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
    
    torch.backends.cudnn.enabled = not deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    
    #Load, parse and save data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    #write_fasta(os.path.join(saved_root,'train.fasta'),train_data[0])
    #write_fasta(os.path.join(saved_root,'val.fasta'),val_data[0])
    #test_data = None    
    if test_data_path is not None:
        test_data = load_data(test_data_path)
    #    write_fasta(os.path.join(saved_root,'test.fasta'),test_data[0])
    #data = train_data, val_data, test_data
    #data_path = os.path.join(saved_root,"data.h5")
    #dd.io.save(data_path,data)
    
    #Verify path exist
    paths = [train_answer_path,val_answer_path,
             test_answer_path,region_table_path,
            # train_region_path,val_region_path,
#             test_region_path
            ]

    for path in paths:
        if path is not None and not os.path.exists(path):
            raise Exception("{} is not exist".format(path))
    
    copied_model,copied_executor = get_model_executor(model_config_path,executor_config_path,
                                                       frozen_names=frozen_names,
                                                       save_distribution=save_distribution)
    
    print("Check memory")
    #check_max_memory_usgae(saved_root,copied_model,copied_executor,train_data,
    #                       val_data,batch_size=batch_size)
    del copied_model
    del copied_executor

    print("Memory is available")
    
    model,executor = get_model_executor(model_config_path,executor_config_path,
                                         frozen_names=frozen_names,
                                         save_distribution=save_distribution)
    
    try:
        train(saved_root,epoch,model,executor,train_data,val_data,
              batch_size=batch_size,deterministic=deterministic,**kwargs)     
    except RuntimeError:
        raise Exception("Something wrong ocuurs in {}".format(saved_root))
        
    #Test
    if not only_train:
        #executor_config = read_json(executor_config_path)
        ann_vec_gff_converter = build_ann_vec_gff_converter(BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP)
        #executor = get_executor(model,set_loss=False,set_optimizer=False,**executor_config)
        test_paths = ['test_on_train','test_on_val']
        data_list = [train_data,val_data]
        answer_paths = [train_answer_path,val_answer_path]
        #region_paths = [train_region_path,val_region_path]

        if test_data_path is not None:
            test_paths.append('test_on_test')
            data_list.append(test_data)
            answer_paths.append(test_answer_path)
            #region_paths.append(test_region_path)
            
        for path,data,answer_path in zip(test_paths,data_list,answer_paths):
            path = os.path.join(saved_root,path)
            create_folder(path)
            test(path,model,executor,data,
                 batch_size=batch_size,
                 ann_vec_gff_converter=ann_vec_gff_converter,
                 region_table_path=region_table_path,
                 answer_gff_path=answer_path,
                # answer_region_path=region_path
                )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config "
                        "build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("--augmentation_max",type=int,default=0)
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("-b","--batch_size",type=int,default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--period",default=5,type=int)
    parser.add_argument("--patient",help="Dafault value is 5. If lower(patient) "
                        "is 'none', then model won't be stopped",
                        type=lambda x: int(x) if x.lower() != 'none' else None,default=5)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-x","--test_data_path",help="Path of testing data")
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--only_train",action='store_true')
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--frozen_names",type=lambda x:x.split(','),default=None)
    parser.add_argument("--monitor_target")
    parser.add_argument("--region_table_path")
    parser.add_argument("--train_answer_path",help='The training answer in gff format')
    parser.add_argument("--val_answer_path",help='The validate answer in gff format')
    parser.add_argument("--test_answer_path",help='The testing answer in gff format')
    parser.add_argument("--deterministic",action="store_true")
    #parser.add_argument("--train_region_path",help='The training region ids')
    #parser.add_argument("--val_region_path",help='The validate answer region ids')
    #parser.add_argument("--test_region_path",help='The testing answer region ids')

    args = parser.parse_args()
                
    #Create folder
    create_folder(args.saved_root)
    copy_path(args.saved_root,sys.argv[0])
    #Save setting
    setting_path = os.path.join(args.saved_root,"main_setting.json")
    setting = vars(args)
    write_json(setting,setting_path)

    kwargs = dict(setting)
    del kwargs['saved_root']
    del kwargs['model_config_path']
    del kwargs['executor_config_path']
    del kwargs['gpu_id']
    
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.model_config_path,
             args.executor_config_path,**kwargs)
