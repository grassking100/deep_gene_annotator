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

def test(output_root,model,executor,data,region_table_path=None,
         answer_gff_path=None,batch_size=None,**kwargs):
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.batch_size = batch_size
    engine.set_root(output_root,with_train=False,with_val=False,create_tensorboard=False)
    #Set callbacks
    singal_handler = engine.get_signal_handler(output_root,prefix='test',
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

def main(trained_root,output_root,data_path,
         deterministic=False,**kwargs):
    create_folder(trained_root)
    backend_deterministic(deterministic)
    batch_size = get_batch_size(trained_root)
    best_model,origin_executor = get_best_model_and_origin_executor(trained_root)
    #if model_weights_path is not None:
    #    best_model.load_state_dict(torch.load(model_weights_path))
        
    data = load_data(data_path)
    test(output_root,best_model,origin_executor,data,batch_size=batch_size,**kwargs)

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-i","--trained_root",help="Root of trained file",required=True)
    parser.add_argument("-d","--data_path",help="Path of data",required=True)
    parser.add_argument("-o","--output_root",help="The path that test result would "
                        "be saved",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--deterministic",action="store_true")
    parser.add_argument("--answer_gff_path",help='The answer in gff format')
    parser.add_argument("--region_table_path",help="The path of region table")
    #parser.add_argument("--model_weights_path",help="The path model weights")
    
    args = parser.parse_args()
    setting = vars(args)
    kwargs = dict(setting)
    del kwargs['gpu_id']
        
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
