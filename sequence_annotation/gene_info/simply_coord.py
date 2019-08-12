import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed
from utils import simply_coord

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",help="input_path",required=True)
    parser.add_argument("-o", "--output_path",help="output_path",required=True)
    args = parser.parse_args()
    try:
        write_bed(simply_coord(read_bed(args.input_path)),args.output_path)
    except:
        raise Exception("{} has wrong format".format(args.input_path))
