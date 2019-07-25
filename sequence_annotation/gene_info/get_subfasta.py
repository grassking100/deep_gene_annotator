import sys
import os
from argparse import ArgumentParser
from utils import read_bed
from sequence_annotation.data_handler.fasta import read_fasta,write_fasta

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",help="input_path",required=True)
    parser.add_argument("-b", "--bed_path",help="bed_path",required=True)
    parser.add_argument("-o", "--output_path",help="output_path",required=True)
    args = parser.parse_args()
    ids = set(list(read_bed(args.bed_path)['id']))
    data_ = read_fasta(args.input_path)
    data = {}
    for id_,seq in data_.items():
        if id_ in ids:
            data[id_] = seq
    write_fasta(args.output_path,data)
