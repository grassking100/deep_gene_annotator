import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed
from utils import get_id_table

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename from RNA id to gene id")
    parser.add_argument("-i", "--input_bed_path",help="Input BED file",required=True)
    parser.add_argument("-t", "--id_convert_path",help="Table about translation between gene id and RNA id",required=True)
    parser.add_argument("-o", "--output_bed_path",help="Onput BED file",required=True)
    args = parser.parse_args()
    
    bed = read_bed(args.input_bed_path).to_dict('record')
    id_convert = get_id_table(args.id_convert_path)
    returned = []
    for item in bed:
        translated = dict(item)
        translated['id'] = id_convert[item['id']]
        returned.append(translated) 

    returned = pd.DataFrame.from_dict(returned)
    write_bed(returned,args.output_bed_path)
