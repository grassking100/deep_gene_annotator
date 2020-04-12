import os, sys
import pandas as pd
sys.path.append(os.path.dirname(__file__) + "/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_fasta, write_fasta

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", help="input_path", required=True)
    parser.add_argument("-d", "--id_path", help="id_path", required=True)
    parser.add_argument("-o",
                        "--output_path",
                        help="output_path",
                        required=True)
    args = parser.parse_args()
    ids = list(pd.read_csv(args.id_path, header=None)[0])
    fasta = read_fasta(args.input_path)
    subfasta = {}
    for id_, seq in fasta.items():
        if id_ in ids:
            subfasta[id_] = seq
    write_fasta(args.output_path, subfasta)
