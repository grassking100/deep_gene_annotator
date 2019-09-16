import os,sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed,read_fai

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program split data")
    parser.add_argument("--region_bed_path",required=True)
    parser.add_argument("--region_rename_table_path",required=True)
    parser.add_argument("--fai_path",required=True)
    parser.add_argument("--splitted_id_root",required=True)
    args = parser.parse_args()
    region = read_bed(args.region_bed_path)
    fai = read_fai(args.fai_path)
    min_chrom = None
    min_length = None
    for chrom,length in fai.items():
        update = False
        if min_chrom is None:
            update = True
        elif min_length > length:
            update = True
        if update:    
            min_chrom = str(chrom)
            min_length = length

    region['length'] = region['end'] - region['start'] + 1
    train_val_chrom_ids = [str(chrom) for chrom in fai.keys() if chrom != min_chrom]
    train_val_length = list(region[region['chr'].isin(train_val_chrom_ids)]['length'])
    threshold = int(np.median(train_val_length)*2)
    path = os.path.join(args.splitted_id_root,'threshold.txt')
    with open(path,'w') as fp:
        fp.write(str(threshold))
    
    region_rename_table = pd.read_csv(args.region_rename_table_path,sep='\t')
    train_val_chrom_ids.sort()
    for val_chrom in train_val_chrom_ids:
        train_chroms = [train_chrom for train_chrom in train_val_chrom_ids if train_chrom != val_chrom]
        train = region_rename_table[region_rename_table['chr'].isin(train_chroms)]['new_id']
        val = region_rename_table[region_rename_table['chr'].isin([val_chrom])]['new_id']
        train = train.drop_duplicates()
        val = val.drop_duplicates()
        train_chroms_str = '_'.join([str(chrom) for chrom in train_chroms])
        train_path = os.path.join(args.splitted_id_root,'chrom_{}_new_id.tsv'.format(train_chroms_str))
        val_path = os.path.join(args.splitted_id_root,'chrom_{}_new_id.tsv'.format(val_chrom))
        train.to_csv(train_path,header=False,index=None)
        val.to_csv(val_path,header=False,index=None)

    test = region_rename_table[region_rename_table['chr'].isin([min_chrom])]['new_id']
    test = test.drop_duplicates()
    test_path = os.path.join(args.splitted_id_root,'test_chrom_{}_new_id.tsv'.format(min_chrom))
    test.to_csv(test_path,header=False,index=None)