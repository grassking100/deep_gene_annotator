import os
import sys
import deepdish as dd
from argparse import ArgumentParser
sys.path.append("/home/sequence_annotation")
from main.utils import load_data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_path",help="Parh to save file",required=True)
    parser.add_argument("-i","--id_path",help="Path of ids",required=True)
    parser.add_argument("--max_len",type=int,default=-1,help="Sequences' max length," +\
                        " if it is negative then it will be ignored")
    parser.add_argument("--min_len",type=int,default=0,help="Sequences' min length")
    parser.add_argument("--ratio",type=float,default=1,help="Ratio of number to be chosen to train" +\
                        " and validate, start chosen by increasing order)")
    parser.add_argument("--site_ann_method")
    args = parser.parse_args()
    
    print("Load and parse data")
    if os.path.exists(args.saved_path):
        print("Data is existed, the program will be skipped")
    else:
        data = load_data(args.fasta_path,args.ann_seqs_path,
                         [args.id_path],
                         min_len=args.min_len,
                         max_len=args.max_len,
                         ratio=args.ratio,
                         site_ann_method=args.site_ann_method)[0]
        dd.io.save(args.saved_path,data)
