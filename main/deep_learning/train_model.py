import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import copy_path, create_folder, read_json, write_json
from sequence_annotation.utils.utils import BASIC_COLOR_SETTING, BASIC_GENE_ANN_TYPES,get_time_str
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from sequence_annotation.genome_handler.select_data import load_data
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.process.director import create_model_exe_builder, Trainer
from main.deep_learning.test_model import test

def _get_first_large_data(data, batch_size=None):
    batch_size = batch_size or 1
    seqs, ann_container = data
    lengths = {key: len(seq) for key, seq in seqs.items()}
    print("Max length is {}".format(max(lengths.values())))
    sorted_length_keys = sorted(lengths, key=lengths.get, reverse=True)
    part_keys = sorted_length_keys[:batch_size]
    part_seqs = dict(zip(part_keys, [seqs[key] for key in part_keys]))
    part_container = AnnSeqContainer()
    part_container.ANN_TYPES = ann_container.ANN_TYPES
    for key in part_keys:
        part_container.add(ann_container.get(key))
    return part_seqs, part_container


def _get_max_target_seqs(seqs,ann_seqs,seq_fig_target=None):
    max_count = 0
    selected_id = None
    for ann_seq in ann_seqs:
        count = class_count(ann_seq)[seq_fig_target or 'intron']
        if max_count <= count:
            max_count = count
            selected_id = ann_seq.id
    return seqs[selected_id],ann_seqs[selected_id]


def train(model,train_data,val_data,output_root=None,**kwargs):
    train_seqs, train_anns = train_data
    val_seqs, val_anns = val_data
    seq,ann_seq = _get_max_target_seqs(val_seqs,val_ann_seqs)
    seq_fig = engine.get_seq_fig(seq,ann_seq,color_settings=BASIC_COLOR_SETTING)
    other_callbacks.add(seq_fig)
    #Prcoess data
    raw_data = {
        'train': {'inputs': train_seqs,'answers': train_anns},
        'val':{'inputs': val_seqs, 'answers': val_anns}
    }
    data,stats = AnnSeqProcessor(BASIC_GENE_ANN_TYPES).process(raw_data,True)
    if output_root is not None:
        settings_path = os.path.join(output_root,'settings')
        create_folder(settings_path)
        write_json(stats,os.path.join(settings_path,'train_val_data_stats.json'))
    trainer = Trainer(BASIC_GENE_ANN_TYPES,model,data['train'],
                      data['val'],root=output_root,**kwargs)
    trainer.execute()
    return trainer

def check_max_memory_usage(model,train_data, val_data,output_root=None,exe_builder=None):
    train_batch_size = val_batch_size = None
    if exe_builder is not None:
        train_batch_size = exe_builder.get_data_generator('train').batch_size
        val_batch_size = exe_builder.get_data_generator('test').batch_size
    train_data = _get_first_large_data(train_data, train_batch_size)
    val_data = _get_first_large_data(val_data, val_batch_size)
    try:
        torch.cuda.reset_max_memory_cached()
        train(model,train_data,val_data,executor_builder=exe_builder)
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
         train_data_path,val_data_path,region_table_path,save_distribution=False,
         model_weights_path=None,executor_weights_path=None,**kwargs):
    
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
    
    resource_path = os.path.join(output_root,'resource')
    create_folder(resource_path)
    for path in copied_paths:
        copy_path(resource_path,path)
    
    
    #Verify path exist
    #if not os.path.exists(region_table_path):
    #    raise Exception("{} is not exist".format(region_table_path))
    
    temp_model,temp_exe_builder = create_model_exe_builder(model_settings_path,executor_settings_path,
                                                        save_distribution=save_distribution)
    
    print("Check memory")
    check_max_memory_usage(temp_model,train_data,val_data,
                           output_root=output_root,
                           exe_builder=temp_exe_builder)
    del temp_model
    del temp_exe_builder
    torch.cuda.empty_cache()
    print("Memory is available")
    model,exe_builder = create_model_exe_builder(model_settings_path,executor_settings_path,
                                              save_distribution=save_distribution,
                                              model_weights_path=model_weights_path,
                                              executor_weights_path=executor_weights_path)
    try:
        train(model,train_data,val_data,output_root=output_root,
              executor_builder=exe_builder,**kwargs)     
    except RuntimeError:
        raise Exception("Something wrong occurs in {}".format(output_root))
        
    #Test
    test_paths = ['test_on_train','test_on_val']
    data_list = [train_data,val_data]

    for path,data in zip(test_paths,data_list):
        test_root = os.path.join(output_root,path)
        create_folder(test_root)
        test(model,data,
             output_root=test_root,
             #region_table_path=region_table_path,
             executor_builder=exe_builder)

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
    args = parser.parse_args()
    kwargs = dict(vars(args))
    del kwargs['gpu_id']
    
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
