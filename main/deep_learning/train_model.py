import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import copy_path, create_folder, read_json, write_json
from sequence_annotation.utils.utils import get_time_str
from sequence_annotation.file_process.utils import BASIC_GENE_ANN_TYPES
from sequence_annotation.preprocess.select_data import load_data
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.process.director import create_model_exe_builder, Trainer
from sequence_annotation.process.executor import BasicExecutor
from sequence_annotation.process.checkpoint import build_checkpoint
from sequence_annotation.process.lr_scheduler import LearningRateHolder
from main.deep_learning.test_model import test


def _get_first_large_data(data, batch_size):
    lengths = data['length']#,data['answer'],data['length'],data['id']
    print("Max length is {}".format(max(lengths)))
    part_sorted_indice = np.argsort(lengths)[:batch_size]
    data_ = {}
    for key,values in data.items():
        data_[key] = [values[index] for index in part_sorted_indice]
    return data_

def _get_max_intron_containing_seq(data,ann_types):
    max_count = 0
    selected_index = None
    intron_index = ann_types.index('intron')
    for index,answer in enumerate(data['answer']):
        introns = np.array(answer)[intron_index]
        count = sum(introns)
        if max_count <= count:
            max_count = count
            selected_index = index
    data_ = {}
    for key,values in data.items():
        data_[key] = [values[selected_index]]
    return data_

def train(exe_builder,model,train_data,val_data,output_root=None,visual_data=None,**kwargs):
    train_executor = exe_builder.build('train',model,train_data)
    test_executor = exe_builder.build('test',model,val_data)
    other_executor = BasicExecutor()
    train_executor.callbacks.add(LearningRateHolder(train_executor.optimizer))
    if output_root is not None:
        create_folder(output_root)
        checkpoint_root = os.path.join(output_root, 'checkpoint')
        seq_fig_root = os.path.join(output_root,"seq_fig")
        checkpoint = build_checkpoint(checkpoint_root,'train',model,train_executor)
        other_executor.callbacks.add(checkpoint)
        if visual_data is not None:
            seq_fig = SeqFigCallback(model,visual_data,seq_fig_root)
            other_executor.callbacks.add(seq_fig)
        
    trainer = Trainer(train_executor,test_executor,
                      other_executor,root=output_root,**kwargs)
    trainer.execute()
    return trainer


def check_max_memory_usage(exe_builder,model,train_data,val_data,output_root=None,**kwargs):
    try:
        torch.cuda.reset_max_memory_cached()
        train(exe_builder,model,train_data,val_data,output_root=output_root,**kwargs)
        max_memory = torch.cuda.max_memory_reserved()
        messenge = "Max memory allocated is {}\n".format(max_memory)
        print(messenge)
        if output_root is not None:
            path = os.path.join(output_root, 'max_memory.txt')
            with open(path, "w") as fp:
                fp.write(messenge)
    except RuntimeError:
        if output_root is not None:
            path = os.path.join(output_root, 'error.txt')
            with open(path, "a") as fp:
                fp.write("Memory might be fulled at {}\n".format(get_time_str()))
        raise Exception("Memory is fulled")

        
def main(output_root,model_settings_path,executor_settings_path,
         train_data_path,val_data_path,region_table_path=None,save_distribution=False,
         model_weights_path=None,executor_weights_path=None,
         train_answer_gff_path=None,val_answer_gff_path=None,**kwargs):
    setting = locals()
    kwargs = setting['kwargs']
    del setting['kwargs']
    setting.update(kwargs)
    #Create folder
    create_folder(output_root)
    #Save setting
    setting_path = os.path.join(output_root,"train_main_setting.json")
    if os.path.exists(setting_path):
        existed = read_json(setting_path)
        if setting != existed:
            raise Exception("The {} is not same as previous one".format(setting_path))
    write_json(setting,setting_path)
    
    copied_paths = [model_settings_path,executor_settings_path,train_data_path,val_data_path]
    if model_weights_path is not None:
        copied_paths.append(model_weights_path)

    #Load, parse and save data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    train_seqs, train_anns = train_data
    val_seqs, val_anns = val_data
    resource_path = os.path.join(output_root,'resource')
    create_folder(resource_path)
    for path in copied_paths:
        copy_path(resource_path,path)
    #Prcoess data
    data = {}
    stats = {}
    ann_seq_processor = AnnSeqProcessor(BASIC_GENE_ANN_TYPES)
    for type_,data_ in zip(['training','validation'],[train_data,val_data]):
        data[type_],stats[type_] = ann_seq_processor.process({'seq': data_[0],'answer': data_[1]})
    
    max_intron_containing_data = _get_max_intron_containing_seq(data["validation"],BASIC_GENE_ANN_TYPES)
    
    if output_root is not None:
        settings_path = os.path.join(output_root,'settings')
        create_folder(settings_path)
        write_json(stats,os.path.join(settings_path,'train_val_data_stats.json'))
    
    t_model,t_exe_builder = create_model_exe_builder(model_settings_path,executor_settings_path,
                                                     save_distribution=save_distribution)
    
    train_batch_size = t_exe_builder.get_data_generator('train').batch_size
    val_batch_size = t_exe_builder.get_data_generator('test').batch_size
    fisrt_large_train_data = _get_first_large_data(data['training'], train_batch_size)
    fisrt_large_val_data = _get_first_large_data(data['validation'], val_batch_size)
    print("Check memory")
    check_max_memory_usage(t_exe_builder,t_model,fisrt_large_train_data,fisrt_large_val_data,
                           output_root=output_root,**kwargs)
    del t_model
    torch.cuda.empty_cache()
    print("Memory is available")
    model,exe_builder = create_model_exe_builder(model_settings_path,executor_settings_path,
                                              save_distribution=save_distribution,
                                              model_weights_path=model_weights_path,
                                              executor_weights_path=executor_weights_path)
    

    try:
        train(exe_builder,model,data['training'],data['validation'],
              output_root=output_root,**kwargs)
    
    except RuntimeError:
        raise Exception("Something wrong occurs in {}".format(output_root))
        
    #Test
    test_paths = ['test_on_train','test_on_val']
    data_list = [train_data,val_data]
    answer_gff_paths = [train_answer_gff_path,val_answer_gff_path]

    for path,data,answer_gff_path in zip(test_paths,data_list,answer_gff_paths):
        test_root = os.path.join(output_root,path)
        create_folder(test_root)
        test(exe_builder,model,data,output_root=test_root,
             region_table_path=region_table_path,answer_gff_path=answer_gff_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_settings_path",help="Path of model settings "
                        "build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_settings_path",help="Path of Executor settings",required=True)
    parser.add_argument("-c","--generator_settings",help="Path of Generator settings",required=True)
    parser.add_argument("-s","--output_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("-r","--region_table_path",help="The path of region data table",required=True)
    parser.add_argument("-p","--patience",help="The epoch to stop traininig when val_loss "
                        "is not improving. Dafault value is None, the model won't be "
                        "stopped by early stopping",type=int,default=None)
    parser.add_argument("-n","--epoch",type=int,default=100)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--model_weights_path")
    parser.add_argument("--executor_weights_path")
    parser.add_argument("--train_answer_gff_path")
    parser.add_argument("--val_answer_gff_path")
    
    args = parser.parse_args()
    kwargs = dict(vars(args))
    del kwargs['gpu_id']
    
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
