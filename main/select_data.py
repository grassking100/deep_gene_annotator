import os
import sys
import deepdish as dd
from argparse import ArgumentParser
sys.path.append(os.path.abspath(os.path.dirname(__file__)+"/.."))
from main.utils import select_data
from sequence_annotation.utils.utils import read_gff,write_gff

def main(saved_path,fasta_path,ann_seqs_path,id_path,min_len,max_len,ratio,select_each_type,
         input_gff_path=None,saved_gff_path=None):
    print("Load and parse data")
    if os.path.exists(saved_path):
        data = dd.io.load(saved_path)
        print("Data is existed, the program will be skipped")
        print("Number of parsed data:{}".format(len(data[0])))
    else:
        data = select_data(fasta_path,ann_seqs_path,id_path,
                           min_len=min_len,max_len=max_len,ratio=ratio,
                           select_each_type=select_each_type)
        print("Number of parsed data:{}".format(len(data[0])))
        dd.io.save(saved_path,data)
        print("Save file to {}".format(saved_path))
        
    if not os.path.exists(saved_gff_path) and input_gff_path is not None:
        if saved_gff_path is None:
            raise Exception("Missing saved gff path")
        region_ids = list(data[0].keys())
        gff = read_gff(input_gff_path)
        selected_gff = gff[gff['chr'].isin(region_ids)]
        write_gff(selected_gff,saved_gff_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-a","--ann_seqs_path",help="Path of AnnSeqContainer",required=True)
    parser.add_argument("-s","--saved_path",help="Path to save file",required=True)
    parser.add_argument("-i","--id_path",help="Path of ids",required=True)
    parser.add_argument("--max_len",type=int,default=None,help="Sequences' max length," +\
                        " if it is None then it will be ignored")
    parser.add_argument("--min_len",type=int,default=0,help="Sequences' min length")
    parser.add_argument("--ratio",type=float,default=1,help="Ratio of number to be chosen to parsed" +\
                        ", start from by increasing order)")
    parser.add_argument("--select_each_type",action='store_true',help="The sequences would be separately chosen by each type")
    parser.add_argument("--input_gff_path",help='The answer in gff format')
    parser.add_argument("--saved_gff_path",help="Path to save selected answer in GFF")

    args = parser.parse_args()
    setting = vars(args)
    main(**setting)
