import os
from argparse import ArgumentParser

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config",help="Model config build by SeqAnnBuilder",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-d","--data_path",help="Path of data",required=True)
    parser.add_argument("-w","--model_weights_path",required=True)
    parser.add_argument("-g","--gpu_id",type=str,default=0,help="GPU to used")
    parser.add_argument("--max_len",type=int,default=-1,help="Sequences' max length, if it is negative then it will be ignored")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--use_naive",action="store_true")
    parser.add_argument("--fix_boundary",action="store_true")
    parser.add_argument("--use_seqlogo",action="store_true")
    parser.add_argument("--map_order_config_path")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id
    
import sys
import json
import deepdish as dd
import torch
torch.backends.cudnn.benchmark = True
sys.path.append("/home/sequence_annotation")
from sequence_annotation.genome_handler.load_data import load_data
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.inference import seq_ann_inference
from sequence_annotation.process.callback import SeqLogo
from sequence_annotation.utils.utils import create_folder
from main.utils import load_data, get_model, get_executor, GENE_MAP,ANN_TYPES
    
def test(model,executor,data,batch_size=None,saved_root=None,
         fix_boundary=False,use_seqlogo=False,gene_map=None,
         channel_order=None,ann_types=None,**kwargs):
    channel_order = channel_order or list(data[1].ANN_TYPES)
    engine = SeqAnnEngine(ann_types=ann_types or ANN_TYPES,channel_order=channel_order)
    if saved_root is not None:
        engine.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)
    engine.executor = executor
    engine.fix_boundary = fix_boundary
    engine.gene_map = gene_map or GENE_MAP
    engine.test_seqs,engine.test_ann_seqs = data
    if use_seqlogo:
        seqlogo = SeqLogo('post_act_x_',saved_root,radius=128)
        engine.other_callbacks.add(seqlogo)
        model = model.feature_block
    worker = engine.test(model,batch_size=batch_size,use_default_callback=not use_seqlogo)
    return worker

def main(data_path,model_config_path,model_weights_path,use_naive=True,**kwargs):
    model = get_model(model_config_path,model_weights_path=model_weights_path)
    executor = get_executor(model,use_naive=use_naive,set_loss=False,set_optimizer=False)
    data = dd.io.load(data_path)
    worker = test(model,executor,data,**kwargs)
    return worker

if __name__ == '__main__':
    create_folder(args.saved_root)

    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"test_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(setting, fp, indent=4)
    
    map_order_config = {}
    if args.map_order_config_path is not None:
        with open(args.map_order_config_path,"r") as fp:
            map_order_config = json.load(fp)
    
    main(args.data_path,args.model_config,args.model_weights_path,
         use_naive=args.use_naive,batch_size=args.batch_size,
         saved_root=args.saved_root,fix_boundary=args.fix_boundary,
         use_seqlogo=args.use_seqlogo,**map_order_config)
