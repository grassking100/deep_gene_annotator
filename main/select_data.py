import os
import sys
import deepdish as dd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from main.utils import select_data

def main(saved_path,fasta_path,ann_seqs_path,id_path,min_len,max_len,ratio,select_each_type):
    print("Load and parse data")
    if os.path.exists(saved_path):
        data = dd.io.load(saved_path)
        if isinstance(data[1],dict):
            print("Data is existed, the program will be skipped")
        else:
            data = data[0],data[1].to_dict()
            dd.io.save(saved_path,data)
            print("Save file to {}".format(saved_path))
    else:
        data = select_data(fasta_path,ann_seqs_path,[id_path],
                           min_len=min_len,max_len=max_len,ratio=ratio,
                           select_each_type=select_each_type)
        print("Number of parsed data:{}".format(len(data[0])))
        dd.io.save(saved_path,data)
        print("Save file to {}".format(saved_path))

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
    parser.add_argument("--select_each_type",action='store_true')

    args = parser.parse_args()
    setting = vars(args)
    main(**setting)
