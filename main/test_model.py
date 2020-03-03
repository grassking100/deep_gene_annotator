import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from sequence_annotation.utils.utils import create_folder, write_json,read_json
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES, BASIC_GENE_MAP
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.convert_signal_to_gff import build_ann_vec_gff_converter
from main.utils import get_model_executor,load_data,backend_deterministic

def _get_name(path):
    return path.split('/')[-1]

def test(saved_root,model,executor,data,batch_size=None,
         ann_vec_gff_converter=None,**kwargs):
    
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)

    worker = engine.test(model,executor,data,
                         batch_size=batch_size,
                         ann_vec_gff_converter=ann_vec_gff_converter,**kwargs)
    return worker

def main(saved_root,test_root,data_path,deterministic=False,**kwargs):

    backend_deterministic(deterministic)
    setting = read_json(os.path.join(saved_root,'main_setting.json'))
    batch_size = setting['batch_size']
    executor_config_path = os.path.join(saved_root,_get_name(setting['executor_config_path']))
    model_config_path = os.path.join(saved_root,_get_name(setting['model_config_path']))
    model_weights_path = read_json(os.path.join(saved_root,'best_model.status'))['path']
    model,executor = get_model_executor(model_config_path,executor_config_path,
                                        model_weights_path=model_weights_path)

    data = load_data(data_path)
    converter = build_ann_vec_gff_converter(BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP)
    test(test_root,model,executor,data,ann_vec_gff_converter=converter,
         batch_size=batch_size,**kwargs)

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-s","--saved_root",help="Root of saved file",required=True)
    parser.add_argument("-d","--data_path",help="Path of data",required=True)
    parser.add_argument("-t","--test_root",help='The path that test result would be saved',
                        required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--deterministic",action="store_true")

    args = parser.parse_args()
    
    create_folder(args.saved_root)
    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"test_setting.json")

    if os.path.exists(setting_path):
        existed = read_json(setting_path)
        setting_ = dict(setting)
        del existed['gpu_id']
        del setting_['gpu_id']
        if setting_ != existed:
            raise Exception("The {} is not same as previous one".format(setting_path))
    else:
        write_json(setting,setting_path)
    
    kwargs = dict(setting)
    del kwargs['data_path']
    del kwargs['gpu_id']
    del kwargs['saved_root']
    del kwargs['test_root']
        
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.test_root,args.data_path,**kwargs)
