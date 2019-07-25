import os, sys
import pandas as pd
from argparse import ArgumentParser
from utils import read_bed,write_bed

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will redefine coordinate based on region file")
    parser.add_argument("-i", "--bed_path",help="Bed file to be redefined coordinate",required=True)
    parser.add_argument("-t", "--table_path",help="Table contained region's containing transcript ids",required=True)
    parser.add_argument("-r", "--region_path",help="Bed file to be used as coordinate reference",required=True)
    parser.add_argument("-o", "--output_path",help="Path to saved redefined bed file",required=True)
    args = parser.parse_args()
    bed = read_bed(args.bed_path).to_dict('record')
    region = read_bed(args.region_path).to_dict('record')
    table = pd.read_csv(args.table_path,sep='\t').to_dict('record')
    starts = dict()
    for item in region:
        starts[item['id']] = item['start']
        
    belong_table = {}
    for item in table:
        ids = item['old_id'].split(",")
        for id_ in ids:
            belong_table[id_] = item['new_id']
    
    redefined_bed = []
    for index,item in enumerate(bed):
        bed_item = dict(item)
        belong_id = belong_table[item['id']]
        start = starts[belong_id] - 1
        bed_item['start'] -= start
        bed_item['end'] -= start
        bed_item['thick_start'] -= start
        bed_item['thick_end'] -= start
        bed_item['chr'] = belong_id
        redefined_bed += [bed_item]

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    write_bed(redefined_bed,args.output_path)