import sys
import os
from argparse import ArgumentParser
from utils import read_bed, write_bed, simply_coord

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",help="input_path",required=True)
    parser.add_argument("-o", "--output_path",help="output_path",required=True)
    args = parser.parse_args()
    write_bed(simply_coord(read_bed(args.input_path)),args.output_path)
