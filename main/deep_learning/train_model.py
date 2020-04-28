import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/..")
from sequence_annotation.utils.utils import create_folder, write_json,copy_path,read_json
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine,check_max_memory_usgae,get_model_executor
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.process.utils import BASIC_COLOR_SETTING
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from sequence_annotation.genome_handler.select_data import load_data
from main.utils import backend_deterministic
from main.deep_learning.test_model import test

def _get_max_target_seqs(seqs,ann_seqs,seq_fig_target=None):
    max_count = 0
    selected_id = None
    for ann_seq in ann_seqs:
        count = class_count(ann_seq)[seq_fig_target or 'intron']
        if max_count <= count:
            max_count = count
            selected_id = ann_seq.id
    return seqs[selected_id],ann_seqs[selected_id]

def train(saved_root,epoch,model,executor,train_data,val_data,
          batch_size=None,patient=None,monitor_target=None,period=None,
          discard_ratio_min=None,discard_ratio_max=None,
          augment_up_max=None,augment_down_max=None,
          deterministic=False,other_callbacks=None):
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES,shuffle_train_data=not deterministic)    
    engine.set_root(saved_root,with_test=False,with_train=True,with_val=True)
    other_callbacks = other_callbacks or Callbacks()
    seq,ann_seq = _get_max_target_seqs(val_data[0],val_data[1])
    seq_fig = engine.get_seq_fig(seq,ann_seq,color_settings=BASIC_COLOR_SETTING)
    other_callbacks.add(seq_fig)
    checkpoint_kwargs={'monitor_target':monitor_target,'patient':patient,'period':period}
    seq_collate_fn_kwargs={'discard_ratio_min':discard_ratio_min,
                           'discard_ratio_max':discard_ratio_max,
                           'augment_up_max':augment_up_max,
                           'augment_down_max':augment_down_max}
    worker = engine.train(model,executor,train_data,val_data=val_data,
                          batch_size=batch_size,epoch=epoch,
                          other_callbacks=other_callbacks,
                          seq_collate_fn_kwargs=seq_collate_fn_kwargs,
                          checkpoint_kwargs=checkpoint_kwargs)
    return worker

def main(saved_root,model_config_path,executor_config_path,
         train_data_path,val_data_path,batch_size=None,epoch=None,
         frozen_names=None,save_distribution=False,
         only_train=False,region_table_path=None,
         deterministic=False,**kwargs):
    setting = locals()
    kwargs = setting['kwargs']
    del setting['kwargs']
    setting.update(kwargs)
    #Create folder
    create_folder(saved_root)
    #Save setting
    setting_path = os.path.join(saved_root,"main_setting.json")
    if os.path.exists(setting_path):
        existed = read_json(setting_path)
        if setting != existed:
            raise Exception("The {} is not same as previous one".format(setting_path))
    else:
        write_json(setting,setting_path)
    
    copied_paths = [model_config_path,executor_config_path,train_data_path,val_data_path]

    resource_backup_path = os.path.join(saved_root,'resource')
    create_folder(resource_backup_path)
    for path in copied_paths:
        copy_path(resource_backup_path,path)
    
    backend_deterministic(deterministic)
    
    #Load, parse and save data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    
    #Verify path exist
    if not only_train:
        if region_table_path is not None and not os.path.exists(region_table_path):
            raise Exception("{} is not exist".format(region_table_path))
    
    temp_model,temp_executor = get_model_executor(model_config_path,executor_config_path,
                                                  frozen_names=frozen_names,
                                                  save_distribution=save_distribution)
    
    print("Check memory")
    check_max_memory_usgae(saved_root,temp_model,temp_executor,train_data,
                           val_data,batch_size=batch_size)
    del temp_model
    del temp_executor
    torch.cuda.empty_cache()
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
        test_paths = []
        data_list = []
        
        test_paths.append('test_on_train')
        data_list.append(train_data)

        test_paths.append('test_on_val')
        data_list.append(val_data)
            
        for path,data in zip(test_paths,data_list):
            test_root = os.path.join(saved_root,path)
            create_folder(test_root)
            test(test_root,model,executor,data,batch_size=batch_size,
                 region_table_path=region_table_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config "
                        "build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-b","--batch_size",type=int,default=32)
    parser.add_argument("--augment_up_max",type=int,default=0)
    parser.add_argument("--augment_down_max",type=int,default=0)
    parser.add_argument("--discard_ratio_min",type=float,default=0)
    parser.add_argument("--discard_ratio_max",type=float,default=0)
    parser.add_argument("-n","--epoch",type=int,default=100)
    parser.add_argument("-p","--period",default=1,type=int)
    parser.add_argument("--patient",help="The epoch to stop traininig when monitor_target "
                        "is not improving. Dafault value is None, the model won't be "
                        "stopped by early stopping",type=int,default=None)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--only_train",action='store_true')
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--frozen_names",type=lambda x:x.split(','),default=None)
    parser.add_argument("--monitor_target")
    parser.add_argument("--deterministic",action="store_true")
    parser.add_argument("--region_table_path",help="The path of region data table which its "
                        "old_id is single-strand data's chromosome and new_id is "
                        "double-strand data's chromosome")
    
    args = parser.parse_args()
    kwargs = dict(vars(args))
    del kwargs['saved_root']
    del kwargs['model_config_path']
    del kwargs['executor_config_path']
    del kwargs['gpu_id']
    
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.model_config_path,
             args.executor_config_path,**kwargs)
