import os,sys
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_gff,write_gff
from sequence_annotation.utils.utils import dupliacte_gff_by_parent,get_gff_with_attribute

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    parser.add_argument("--query_column",type=str,default='id',required=True)
    parser.add_argument("--threshold",type=float,required=True)
    parser.add_argument("--mode",type=int,help='Value selected should be 0) larger than or 1) smaller than'+
                        'or 2) equal or 3) larger than or equal or 4) smaller than or equal threshold',required=True)
    
    args = parser.parse_args()
    gff = read_gff(args.input_path)
    gff = get_gff_with_attribute(gff,split_attr=['parent'])
    gff = dupliacte_gff_by_parent(gff)
    column = gff[args.query_column].astype(float)
    if args.mode == 0:
        match_index = column > args.threshold 
    elif args.mode == 1:    
        match_index = column < args.threshold 
    elif args.mode == 2:    
        match_index = column == args.threshold
    elif args.mode == 3:    
        match_index = column >= args.threshold
    elif args.mode == 4:    
        match_index = column <= args.threshold   
    else:
        raise Exception("Wrong mode")
    
    selected = gff[match_index]
    parents = set(list(selected['parent']))
    ids = set(list(selected['id']))
    selected = gff[(gff['parent'].isin(parents)) | (gff['parent'].isin(ids)) | (gff['id'].isin(parents)) | (gff['id'].isin(ids))]
    write_gff(selected,args.output_path)
