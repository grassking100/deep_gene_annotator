import os, sys
sys.path.append(os.path.dirname(__file__))
from utils import read_bed
import os
import pandas as pd
from argparse import ArgumentParser

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    parser.add_argument("-b", "--bed_path",
                        help="consist_data_path",required=True)
    args = parser.parse_args()

    bed = read_bed(args.bed_path)
    record = bed.to_dict('record')
    new_df = []
    for item in record:
        temp = {}
        temp['gene_id'] = item['chr']+"_"+item['strand']+"_"+str(item['start'])+"_"+str(item['end'])
        temp['transcript_id'] = item['id']
        new_df += [temp]
    new_df = pd.DataFrame.from_dict(new_df)
    new_df.to_csv(os.path.join(args.saved_root,"id_table.tsv"),index=None,sep='\t')
