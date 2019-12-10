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
    parser.add_argument("--use_gffcompare",action="store_true")
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
from sequence_annotation.genome_handler.seq_container import EmptyContainerException
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.inference import seq_ann_inference
from sequence_annotation.process.callback import SeqLogo
from sequence_annotation.process.inference import AnnVec2InfoConverter
from sequence_annotation.utils.utils import create_folder, save_as_gff_and_bed, gffcompare_command, read_gff
from main.utils import load_data, get_model, get_executor,ANN_TYPES,GENE_MAP

def _fix_gff(dna_dict,ann_vec2info_converter,input_gff_path,output_root):
    predict_gff = read_gff(input_gff_path)
    predict_bed_path = os.path.join(output_root,"test_gffcompare_1.bed")
    predict_gff_path = os.path.join(output_root,"test_gffcompare_1.gff3")
    try:
        fixed_train_gff = ann_vec2info_converter.fix_boundary(predict_gff,dna_dict)
        save_as_gff_and_bed(fixed_train_gff,predict_gff_path,predict_bed_path)
    except EmptyContainerException:
        raise 

def _fix_gff_and_compare(origin_path,fixed_path,dna_dict,ann_vec2info_converter):
    input_gff_path = os.path.join(origin_path,'test','test_gffcompare_1.gff3')
    answer_gff_path = os.path.join(origin_path,'test',"answers.gff3")
    predict_gff_path = os.path.join(fixed_path,"test_gffcompare_1.gff3")
    prefix_path = os.path.join(fixed_path,"test_gffcompare_1")
    if os.path.exists(input_gff_path):
        try:
            _fix_gff(dna_dict,ann_vec2info_converter,input_gff_path,fixed_path)
            gffcompare_command(answer_gff_path,predict_gff_path,prefix_path)
        except EmptyContainerException:
            pass

def test(model,executor,data,batch_size=None,saved_root=None,
         use_gffcompare=False,fix_boundary=False,
         ann_vec2info_converter=None,use_seqlogo=False,**kwargs):
    engine = SeqAnnEngine(ann_types=ANN_TYPES)
    if saved_root is not None:
        engine.set_root(saved_root,with_train=False,with_val=False,create_tensorboard=False)
    engine.executor = executor
    engine.use_gffcompare = use_gffcompare
    engine.ann_vec2info_converter = ann_vec2info_converter
    engine.test_seqs,engine.test_ann_seqs = data
    if use_seqlogo:
        seqlogo = SeqLogo('post_act_x_',saved_root,radius=128)
        engine.other_callbacks.add(seqlogo)
        model = model.feature_block
    worker = engine.test(model,batch_size=batch_size,use_default_callback=not use_seqlogo)
    
    if fix_boundary and use_gffcompare and saved_root is not None and ann_vec2info_converter is not None:
        test_by_fixed_path = os.path.join(saved_root,'test_by_fixed')
        create_folder(test_by_fixed_path)
        _fix_gff_and_compare(saved_root,test_by_fixed_path,data[0],ann_vec2info_converter)
    
    return worker

def main(data_path,model_config_path,model_weights_path=None,
         use_naive=True,map_order_config_path=None,**kwargs):
    map_order_config = {}
    if map_order_config_path is not None:
        with open(map_order_config_path,"r") as fp:
            map_order_config = json.load(fp)
    model = get_model(model_config_path,model_weights_path=model_weights_path)
    executor = get_executor(model,use_naive=use_naive,set_loss=False,set_optimizer=False)
    data = dd.io.load(data_path)
    ann_vec2info_converter = AnnVec2InfoConverter(ANN_TYPES,GENE_MAP)
    worker = test(model,executor,data,ann_vec2info_converter=ann_vec2info_converter,
                  **map_order_config,**kwargs)

    return worker

if __name__ == '__main__':
    create_folder(args.saved_root)
    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"test_setting.json")
    with open(setting_path,"w") as fp:
        json.dump(setting, fp, indent=4)

    kwargs = dict(setting)
    del kwargs['data_path']
    del kwargs['model_config']
    del kwargs['gpu_id']
        
    main(args.data_path,args.model_config,**kwargs)
