import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/..")
from sequence_annotation.utils.utils import create_folder
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine,get_best_model_and_origin_executor,get_batch_size
from sequence_annotation.genome_handler.select_data import load_data
from main.utils import backend_deterministic

def test(saved_root,model,executor,data,batch_size=None,**kwargs):
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)
    worker = engine.test(model,executor,data,batch_size=batch_size,**kwargs)
    return worker

def main(saved_root,result_root,data_path,deterministic=False,**kwargs):
    create_folder(saved_root)
    backend_deterministic(deterministic)
    batch_size = get_batch_size(saved_root)
    best_model,origin_executor = get_best_model_and_origin_executor(saved_root)
    data = load_data(data_path)
    test(result_root,best_model,origin_executor,data,batch_size=batch_size,**kwargs)

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-s","--saved_root",help="Root of saved file",required=True)
    parser.add_argument("-d","--data_path",help="Path of data",required=True)
    parser.add_argument("-r","--result_root",help="The path that test result would "
                        "be saved",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--deterministic",action="store_true")
    parser.add_argument("--answer_gff_path",help='The answer in gff format')
    parser.add_argument("--region_table_path",help="The path of region data table "
                        "which its old_id is single-strand data's chromosome and "
                        "new_id is double-strand data's chromosome")
    parser.add_argument("--is_answer_double_strand",action="store_true")
    
    args = parser.parse_args()
    setting = vars(args)
    kwargs = dict(setting)
    del kwargs['data_path']
    del kwargs['gpu_id']
    del kwargs['saved_root']
    del kwargs['result_root']
        
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.result_root,args.data_path,**kwargs)
