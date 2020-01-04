import os
import sys
import json
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import deepdish as dd

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("--saved_root",help="Root to save file",required=True)
    parser.add_argument("--raw_output_path",help="The path of raw output file in h5 format",required=True)
    parser.add_argument("--region_path",help="The path of region data",required=True)
    parser.add_argument("-g","--gpu_id",type=str,default='0',help="GPU to used")
    parser.add_argument("--no_fix",action='store_true')
    parser.add_argument("--transcript_threshold",type=float,default=0.5)
    parser.add_argument("--intron_threshold",type=float,default=0.5)
    parser.add_argument("--length_threshold",type=int,default=0)
    parser.add_argument("--distance",type=int,default=0)
    parser.add_argument("--gene_length_threshold",type=int,default=0)
    parser.add_argument("--use_native",action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =  args.gpu_id

import torch
torch.backends.cudnn.benchmark = True
from keras.preprocessing.sequence import pad_sequences
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.preprocess.flip_coordinate import flip_gff
from sequence_annotation.process.inference import basic_inference,seq_ann_inference,build_converter
from sequence_annotation.process.utils import get_seq_mask
from sequence_annotation.utils.utils import create_folder, write_gff, gffcompare_command, read_gff,print_progress, write_json
from main.utils import ANN_TYPES,GENE_MAP

basic_inference_ = basic_inference()

def convert_outputs_to_tensors(outputs):
    """Convert outputs to Tensors for each attributes"""
    data = {}
    columns = ['dna_seqs', 'lengths', 'outputs','chrom_ids']
    for key in columns:
        data[key] = []
    for index,item in enumerate(outputs):
        print_progress("{}% of data have been processed".format(int(100*index/len(outputs))))
        for key in ['dna_seqs', 'lengths','chrom_ids']:
            data[key] += [item[key]]
        data['outputs'] += list(np.transpose(item['outputs'],(0,2,1)))
    for key in ['dna_seqs', 'lengths','chrom_ids']:
        data[key] = np.concatenate(data[key])
    data['outputs'] = pad_sequences(data['outputs'],padding='post',dtype='float32')
    data['outputs'] = torch.Tensor(data['outputs'].transpose(0,2,1)).cuda()
    data['masks'] = get_seq_mask(data['lengths'],to_cuda=True)
    return data

def inference(chrom_ids,lengths,masks,dna_seqs,ann_vecs,ann_vec2info_converter,
              no_fix=False,use_native=True,
              transcript_threshold=None,intron_threshold=None):
    if use_native:
        ann_vecs = basic_inference_(ann_vecs,masks)
    else:
        ann_vecs = seq_ann_inference(ann_vecs,masks,
                                     transcript_threshold=transcript_threshold,
                                     intron_threshold=intron_threshold)
    ann_vecs = ann_vecs.cpu().numpy()
    if no_fix:
        info = ann_vec2info_converter.vecs2info(chrom_ids,lengths,ann_vecs)
    else:
        info = ann_vec2info_converter.vecs2fixed_info(chrom_ids,lengths,dna_seqs,ann_vecs)
    return info
    
def convert_inference(raw_output_path,**kwargs):
    raw_outputs = dd.io.load(raw_output_path)
    outputs = convert_outputs_to_tensors(raw_outputs)
    gff = inference(outputs['chrom_ids'],outputs['lengths'],outputs['masks'],
                    outputs['dna_seqs'],outputs['outputs'],**kwargs)
    return gff

def main(saved_root,raw_output_path,region_path,
         length_threshold=None,distance=None,
         gene_length_threshold=None,**kwargs):
    predicted_gff_path = os.path.join(saved_root,'predicted.gff')
    prefix_path = os.path.join(saved_root,"gffcompare")
    ann_vec2info_converter = build_converter(ANN_TYPES,GENE_MAP,distance=distance,
                                             length_threshold=length_threshold,
                                             gene_length_threshold=gene_length_threshold)
    config = ann_vec2info_converter.get_config()
    config_path = os.path.join(saved_root,"ann_vec2info_converter_config.json")
    write_json(config,config_path)
    gff = convert_inference(raw_output_path,ann_vec2info_converter=ann_vec2info_converter,**kwargs)
    regions = pd.read_csv(region_path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    redefined_gff = flip_gff(gff,regions)
    write_gff(redefined_gff,predicted_gff_path)

if __name__ == '__main__':    
    create_folder(args.saved_root)
    setting = vars(args)
    setting_path = os.path.join(args.saved_root,"inference_setting.json")
    write_json(setting,setting_path)

    kwargs = dict(setting)
    del kwargs['saved_root']
    del kwargs['raw_output_path']
    del kwargs['gpu_id']
    del kwargs['region_path']
        
    main(args.saved_root,args.raw_output_path,
         args.region_path,**kwargs)
