import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder,write_json
from sequence_annotation.file_process.utils import BASIC_GENE_ANN_TYPES
from sequence_annotation.preprocess.select_data import load_data
from sequence_annotation.process.director import Tester
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.process.signal_handler import create_signal_handler
from sequence_annotation.process.director import create_best_model_exe_builder


def test(executor_builder,model,data,output_root=None,region_table_path=None,answer_gff_path=None,**kwargs):
    annseq_processor = AnnSeqProcessor(BASIC_GENE_ANN_TYPES)
    data_,stats = annseq_processor.process({'seq': data[0],'answer': data[1]})
    test_executor = executor_builder.build('test',model,data_,output_root)
    if output_root is not None:
        settings_path = os.path.join(output_root,'settings')
        create_folder(settings_path)
        write_json(stats,os.path.join(settings_path,'test_stats.json'))
        if region_table_path is not None:
            signal_handler = create_signal_handler(BASIC_GENE_ANN_TYPES,region_table_path,output_root,
                                                   answer_gff_path=answer_gff_path)
            test_executor.callbacks.add(signal_handler)
    tester = Tester(test_executor,root=output_root,**kwargs)
    tester.execute()
    return tester

def main(trained_root,output_root,data_path,**kwargs):
    create_folder(trained_root)
    best_model,origin_executor = create_best_model_exe_builder(trained_root)
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
