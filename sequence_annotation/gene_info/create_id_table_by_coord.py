import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b", "--input_bed_path",required=True)
    parser.add_argument("-t", "--table_path",required=True)
    parser.add_argument("-p", "--prefix",required=True)
    args = parser.parse_args()

    record = read_bed(args.input_bed_path).to_dict('record')
    new_df = []
    index = 0
    id_convert = {}
    for item in record:
        index += 1
        coord_id = "{}_{}_{}_{}".format(item['chr'],item['strand'],item['start'],item['end'])
        id_convert[coord_id] = "{}_{}".format(args.prefix,index)
    
    for item in record:
        coord_id = "{}_{}_{}_{}".format(item['chr'],item['strand'],item['start'],item['end'])
        temp = {}
        temp['gene_id'] = id_convert[coord_id]
        temp['transcript_id'] = item['id']
        new_df += [temp]
    new_df = pd.DataFrame.from_dict(new_df)
    new_df.to_csv(args.table_path,index=None,sep='\t')
    