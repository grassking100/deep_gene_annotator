import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed
from utils import get_id_table

def convert_bed_id(bed,id_convert_table,query):
    returned = []
    for item in bed.to_dict('record'):
        item = dict(item)
        item[query] = id_convert_table[item[query]]
        returned.append(item) 
    returned = pd.DataFrame.from_dict(returned).sort_values(by=['id'])
    return returned

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename by table")
    parser.add_argument("-i", "--input_bed_path",help="Input BED file",required=True)
    parser.add_argument("-t", "--id_convert_path",help="Table about id conversion",required=True)
    parser.add_argument("-o", "--output_bed_path",help="Onput BED file",required=True)
    parser.add_argument("--query",help="Column name to query and replace",default='id')
    args = parser.parse_args()
    
    bed = read_bed(args.input_bed_path)
    id_convert_table = get_id_table(args.id_convert_path)
    returned = convert_bed_id(bed,id_convert_table,args.query)
    write_bed(returned,args.output_bed_path)
