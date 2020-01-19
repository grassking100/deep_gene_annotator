import os
import sys
import torch
import deepdish as dd
from argparse import ArgumentParser
torch.backends.cudnn.benchmark = True
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.utils.utils import create_folder, write_json
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES, BASIC_GENE_MAP
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.callback import SeqLogo,Callbacks
from sequence_annotation.process.convert_singal_to_gff import build_converter
from main.utils import get_executor, get_model,load_data

def test(saved_root,model,executor,data,batch_size=None,
         use_gffcompare=False,ann_vec2info_converter=None,
         use_seqlogo=False,**kwargs):
    
    if use_gffcompare and use_seqlogo:
        raise Exception("The use_gffcompare and use_seqlogo can not set to be True together")
    
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)
    callbacks = Callbacks()
    if use_seqlogo:
        seqlogo = SeqLogo('post_act_x_',saved_root,radius=128)
        callbacks.add(seqlogo)
        model = model.feature_block

    worker = engine.test(model,executor,data,
                         batch_size=batch_size,
                         ann_vec2info_converter=ann_vec2info_converter,
                         callbacks=callbacks,**kwargs)
    return worker

def main(saved_root,data_path,model_config_path,
         model_weights_path=None,use_naive=True,**kwargs):
    model = get_model(model_config_path,model_weights_path=model_weights_path)
    executor = get_executor(model,use_naive=use_naive,set_loss=False,set_optimizer=False)
    data = load_data(data_path)
    converter = build_converter(BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP)
    
    worker = test(saved_root,model,executor,data,
                  ann_vec2info_converter=converter,**kwargs)

    return worker

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config",help="Model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-d","--data_path",help="Path of data",required=True)
    parser.add_argument("-w","--model_weights_path",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--max_len",type=int,default=-1,
                        help="Sequences' max length, if it is negative then it will be ignored")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--use_naive",action="store_true")
    parser.add_argument("--use_gffcompare",action="store_true")
    parser.add_argument("--use_seqlogo",action="store_true")
    parser.add_argument("--region_table_path")

    args = parser.parse_args()
    
    create_folder(args.saved_root)
    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"test_setting.json")
    write_json(setting,setting_path)

    kwargs = dict(setting)
    del kwargs['data_path']
    del kwargs['model_config']
    del kwargs['gpu_id']
    del kwargs['saved_root']
        
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.data_path,args.model_config,**kwargs)
