import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_gff, get_gff_with_attribute
from sequence_annotation.preprocess.utils import RNA_TYPES

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--gff_path",required=True)
    parser.add_argument("-o", "--table_path",required=True)
    args = parser.parse_args()

    record = read_gff(args.gff_path)
    rnas = record[record['feature'].isin(RNA_TYPES)]
    rnas = get_gff_with_attribute(rnas).to_dict('record')
    
    new_df = []
    for rna in rnas:
        temp = {}
        temp['gene_id'] = rna['parent']
        temp['transcript_id'] = rna['id']
        new_df += [temp]

    new_df = pd.DataFrame.from_dict(new_df)
    new_df.to_csv(args.table_path,index=None,sep='\t')
