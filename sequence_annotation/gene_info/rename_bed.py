import os, sys
import pandas as pd
from argparse import ArgumentParser
from utils import read_bed,write_bed

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename bed file")
    parser.add_argument("-i", "--bed_path",help="Bed file to be renamed",required=True)
    parser.add_argument("-p", "--id_prefix",help="Prefix of new id",required=True)
    parser.add_argument("-t", "--saved_table_path",help="Path to saved renamed table",required=True)
    parser.add_argument("-o", "--renamed_bed_path",help="Path to saved renamed bed file",required=True)
    args = parser.parse_args()
    bed = read_bed(args.bed_path).sort_values(by=['chr','start','end','strand']).to_dict('record')
    renamed_table = []
    renamed_bed = []
    index_length = len(str(len(bed)))
    region_id = {}
    
    for index,item in enumerate(bed):
        seq_id = "{prefix}_{index:0>"+str(index_length)+"d}"
        seq_id = seq_id.format(prefix=args.id_prefix,index=index+1)
        region_id[item['id']] = seq_id
        table_item = {}
        for key in ['chr','strand','start','end']:
            table_item[key] = item[key]
        table_item['new_id'] = seq_id
        table_item['old_id'] = item['id']
        renamed_table += [table_item]

    for item in bed:
        renamed_item = dict(item)
        renamed_item['id'] = region_id[item['id']]
        renamed_bed += [renamed_item]

    renamed_table = pd.DataFrame.from_dict(renamed_table).sort_values(by=['new_id'])
    renamed_bed = pd.DataFrame.from_dict(renamed_bed).sort_values(by=['id'])
    renamed_table = renamed_table[['new_id','chr','strand','start','end','old_id']]
    renamed_table.to_csv(args.saved_table_path,index=None,sep='\t')
    write_bed(renamed_bed,args.renamed_bed_path)