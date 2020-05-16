import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine,get_best_model_and_origin_executor,get_batch_size
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.genome_handler.select_data import load_data
from main.utils import backend_deterministic

def test(saved_root,model,executor,data,region_table_path,
         answer_gff_path=None,batch_size=None,**kwargs):
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.batch_size = batch_size
    engine.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)
    #Set callbacks
    singal_handler = engine.get_signal_handler(saved_root,prefix='test',
                                               inference=executor.inference,
                                               region_table_path=region_table_path,
                                               answer_gff_path=answer_gff_path)
    callbacks = Callbacks()
    callbacks.add(singal_handler)
    #Set loader
    test_seqs, test_ann_seqs = data
    raw_data = {'testing': {'inputs': test_seqs, 'answers': test_ann_seqs}}
    data = engine.process_data(raw_data)
    generator = engine.create_basic_data_gen()
    data_loader = generator(data['testing'])
    worker = engine.test(model,executor,data_loader,callbacks=callbacks,**kwargs)
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
    parser.add_argument("--region_table_path",help="The path of region table")
    
    args = parser.parse_args()
    setting = vars(args)
    kwargs = dict(setting)
    del kwargs['data_path']
    del kwargs['gpu_id']
    del kwargs['saved_root']
    del kwargs['result_root']
        
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.result_root,args.data_path,**kwargs)
