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
from main.utils import load_data, get_model, get_executor, GENE_MAP
    
def test(model,executor,data,batch_size=None,saved_root=None,fix_boundary=False):
    engine = SeqAnnEngine()
    if saved_root is not None:
        engine.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)
    engine.executor = executor
    engine.fix_boundary = fix_boundary
    engine.simplify_map = GENE_MAP
    engine.test_seqs,engine.test_ann_seqs = data
    record = engine.test(model,batch_size=batch_size)
    return record

def main(data_path,model_config_path,model_weights_path,use_naive=True,
         batch_size=None,saved_root=None,fix_boundary=False):
    model = get_model(model_config_path,model_weights_path=model_weights_path)
    executor = get_executor(model,use_naive=use_naive,
                            set_loss=False,set_optimizer=False)
    data = dd.io.load(data_path)
    record = test(model,executor,data,batch_size=batch_size,
                  saved_root=saved_root,fix_boundary=fix_boundary)
    return record

if __name__ == '__main__':
    if not os.path.exists(args.saved_root):
        os.mkdir(args.saved_root)

    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"test_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(setting, fp, indent=4)
    
    main(args.data_path,args.model_config,args.model_weights_path,
         use_naive=args.use_naive,batch_size=args.batch_size,
         saved_root=args.saved_root,fix_boundary=args.fix_boundary)
