import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed

def rename_chrom(bed,renamed_table):
    belong_table = {}
    for item in renamed_table.to_dict('record'):
        id_ = item['old_id']
        belong_table[id_] = item['new_id']

    redefined_bed = []
    for item in bed.to_dict('record'):
        bed_item = dict(item)
        bed_item['chr'] = belong_table[item['chr']]
        redefined_bed += [bed_item]

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def main(bed_path,table_path,output_path):
    bed = read_bed(bed_path)
    renamed_table = pd.read_csv(table_path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    redefined_bed = rename_chrom(bed,renamed_table)
    write_bed(redefined_bed,output_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename chromosome")
    parser.add_argument("-i", "--bed_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--table_path",help="Table about renamed region",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)
                        
    args = parser.parse_args()
    main(args.bed_path,args.table_path,args.output_path)
    