import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import read_bed,write_bed
import os
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b", "--bed_path",help="bed_path",required=True)
    parser.add_argument("-s", "--saved_table_root",help="saved_table_root",required=True)
    parser.add_argument("-p", "--name_prefix",help="name_prefix",required=True)
    parser.add_argument("-r", "--renamed_bed_name",help="renamed_bed_name",required=True)
    args = vars(parser.parse_args())
    saved_table_root = args['saved_table_root']
    bed_path = args['bed_path']
    renamed_bed_name = args['renamed_bed_name']
    name_prefix = args['name_prefix']
    bed = read_bed(bed_path).sort_values(by=['chr','start','end','strand'])
    record = bed.to_dict('record')
    new_df = []
    new_bed = []
    index_length = len(str(len(record)))
    for index,item in enumerate(record):
        seq_id = "{prefix}_{index:0>"+str(index_length)+"d}"
        seq_id = seq_id.format(prefix=name_prefix,index=index+1)
        temp = {}
        temp['seq_id'] = seq_id
        temp['contain_transcript_ids'] = item['id']
        temp['chr'] = item['chr']
        temp['start'] = item['start']
        temp['end'] = item['end']
        temp['strand'] = item['strand']
        new_df += [temp]
        new_item = dict(item)
        new_item['id'] = seq_id
        new_bed += [new_item]
    new_df = pd.DataFrame.from_dict(new_df).sort_values(by=['seq_id'])
    new_bed = pd.DataFrame.from_dict(new_bed).sort_values(by=['id'])
    new_df = new_df[['seq_id','chr','strand','start','end','contain_transcript_ids']]
    new_df.to_csv(saved_table_root+"/rename_table.tsv",index=None,sep='\t')
    write_bed(new_bed,saved_table_root+"/"+renamed_bed_name)