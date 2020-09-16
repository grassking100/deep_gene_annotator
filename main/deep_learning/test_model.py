import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder,write_json,BASIC_GENE_ANN_TYPES
from sequence_annotation.process.director import Tester
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.genome_handler.select_data import load_data
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.process.signal_handler import get_signal_handler

def create_best_model_exe_builder(saved_root,latest=False):
    setting = read_json(os.path.join(saved_root, 'train_main_setting.json'))
    resource_root = os.path.join(saved_root,'resource')
    checkpoint_root = os.path.join(saved_root,'checkpoint')
    exe_file_name = get_file_name(setting['executor_settings_path'], True)
    model_file_name = get_file_name(setting['model_settings_path'], True)
    executor_settings_path = os.path.join(resource_root,exe_file_name)
    model_settings_path = os.path.join(resource_root,model_file_name)
    if latest:
        model_weights_path = os.path.join(checkpoint_root,'latest_model.pth')
    else:
        model_weights_path = os.path.join(checkpoint_root,'best_model.pth')
    model, executor = create_model_exe_builder(model_settings_path,executor_settings_path,
                                            model_weights_path=model_weights_path)
    return model, executor


def test(model,data,output_root=None,region_table_path=None,answer_gff_path=None,**kwargs):
    seqs, anns = data
    raw_data = {'test': {'inputs': seqs,'answers': anns}}
    data,stats = AnnSeqProcessor(BASIC_GENE_ANN_TYPES).process(raw_data,True)
    callbacks = Callbacks()
    if output_root is not None:
        settings_path = os.path.join(output_root,'settings')
        create_folder(settings_path)
        write_json(stats,os.path.join(settings_path,'test_stats.json'))
        signal_handler = get_signal_handler(BASIC_GENE_ANN_TYPES,output_root,
                                            region_table_path=region_table_path,
                                            answer_gff_path=answer_gff_path)
        callbacks.add(signal_handler)
    tester = Tester(BASIC_GENE_ANN_TYPES,model,data['test'],
                    root=output_root,callbacks=callbacks,**kwargs)    
    tester.execute()
    return tester

def main(trained_root,output_root,data_path,**kwargs):
    create_folder(trained_root)
    best_model,origin_executor = get_best_model_and_origin_executor(trained_root)
    data = load_data(data_path)
    test(output_root,best_model,data,executor_builder=exe_builder,**kwargs)

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-i","--trained_root",help="Root of trained file",required=True)
    parser.add_argument("-d","--data_path",help="Path of data",required=True)
    parser.add_argument("-o","--output_root",help="The path that test result would "
                        "be saved",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--answer_gff_path",help='The answer in gff format')
    parser.add_argument("--region_table_path",help="The path of region table")
    
    args = parser.parse_args()
    setting = vars(args)
    kwargs = dict(setting)
    del kwargs['gpu_id']
        
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
