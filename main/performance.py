import os
import sys
import json
import pandas as pd
from argparse import ArgumentParser

sys.path.append("/home/sequence_annotation")
from sequence_annotation.utils.utils import read_gff, read_fai, create_folder, write_gff, write_json
from sequence_annotation.process.performance import gff_performance,draw_contagion_matrix
from sequence_annotation.preprocess.utils import get_gff_with_intron

def main(predict_path,answer_path,fai_path,saved_root):
    predict = get_gff_with_intron(read_gff(predict_path))
    answer = get_gff_with_intron(read_gff(answer_path))
    chrom_lengths = read_fai(fai_path)
    
    for chr_ in set(answer['chr']):
        if chr_ not in list(chrom_lengths.keys()):
            raise Exception(chr_,chrom_lengths)
    
    result = gff_performance(predict,answer,chrom_lengths,5)
    base_perform,contagion,block_perform,errors,site_p_a_diff,site_a_p_abs_diff = result
    
    write_json(base_perform,os.path.join(saved_root,'base_performance.json'))
    
    write_json(contagion.tolist(),os.path.join(saved_root,'contagion_matrix.json'))

    write_json(block_perform,os.path.join(saved_root,'block_performance.json'))
        
    write_gff(errors, os.path.join(saved_root,'error_status.gff'))
    
    write_json(site_p_a_diff,os.path.join(saved_root,'p_a_abs_diff.json'))
    
    write_json(site_a_p_abs_diff,os.path.join(saved_root,'a_p_abs_diff.json'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--predict_path",help='The path of prediction result in GFF format',required=True)
    parser.add_argument("--answer_path",help='The path of answer result in GFF format',required=True)
    parser.add_argument("--fai_path",help='The path of chromosome length in fai format',required=True)
    parser.add_argument("--saved_root",help="Path to save result",required=True)
    
    args = parser.parse_args()
    
    create_folder(args.saved_root)
    
    config_path = os.path.join(args.saved_root,'performance_setting.json')
    config = vars(args)
    
    write_json(config,config_path)

    main(args.predict_path,args.answer_path,args.fai_path,args.saved_root)
