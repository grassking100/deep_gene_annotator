import os
import sys
import json
import pandas as pd
from argparse import ArgumentParser

sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import read_gff, read_fai, create_folder, write_gff
from sequence_annotation.process.performance import gff_performance,draw_contagion_matrix

def main(predict_path,answer_path,fai_path,saved_root):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    chrom_lengths = read_fai(fai_path)
    
    for chr_ in set(answer['chr']):
        if chr_ not in list(chrom_lengths.keys()):
            raise Exception(chr_,chrom_lengths)
    
    result = gff_performance(predict,answer,chrom_lengths,3)
    base_performance,contagion_matrix,block_performance,error_status = result
    
    contagion_matrix_path = os.path.join(saved_root,'contagion_matrix.json')
    with open(contagion_matrix_path,"w") as fp:
        json.dump(contagion_matrix.tolist(), fp, indent=4)

    base_performance_path = os.path.join(saved_root,'base_performance.json')
    with open(base_performance_path,"w") as fp:
        json.dump(base_performance, fp, indent=4)

    block_performance_path = os.path.join(saved_root,'block_performance.json')
    with open(block_performance_path,"w") as fp:
        json.dump(block_performance, fp, indent=4)
        
    error_status_path = os.path.join(saved_root,'error_status.gff')
    write_gff(error_status,error_status_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--predict_path",help='The path of prediction result in GFF format',required=True)
    parser.add_argument("--answer_path",help='The path of answer result in GFF format',required=True)
    parser.add_argument("--fai_path",help='The path of chromosome length in fai format',required=True)
    parser.add_argument("--saved_root",help="Path to save result",required=True)
    
    args = parser.parse_args()
    
    create_folder(args.saved_root)
    
    config_path = os.path.join(args.saved_root,'config.json')
    config = vars(args)
    
    with open(config_path,"w") as fp:
        json.dump(config, fp, indent=4)

    main(args.predict_path,args.answer_path,args.fai_path,args.saved_root)
