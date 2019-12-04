import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed
from utils import get_id_table

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will count id in BED file")
    parser.add_argument("-i", "--input_bed_path",help="Input BED file",required=True)
    parser.add_argument("-t", "--id_convert_path")
    args = parser.parse_args()
    
    bed = read_bed(args.input_bed_path).to_dict('record')
    if args.id_convert_path is not None:
        id_convert = get_id_table(args.id_convert_path)
    ids = set()
    for item in bed:
        if args.id_convert_path is not None:
            id_ = id_convert[item['id']]
        else:
            id_ = item['id']
        ids.add(id_)
    print(len(ids))
