import os,sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed,read_fai,write_json

def _get_subtable(table,chrom_ids):
    if not isinstance(chrom_ids,list):
        chrom_ids = [chrom_ids]
    return table[table['chr'].isin(chrom_ids)]

def _get_chrom_str(chrom_ids):
    if not isinstance(chrom_ids,list):
        chrom_ids = [chrom_ids]
        
    chroms_str = '_'.join([str(chrom) for chrom in sorted(chrom_ids)])
    return chroms_str

def write_id_table(region_rename_table,saved_root=None,path=None):
    data = region_rename_table['new_id'].drop_duplicates()
    chrom_ids = sorted(set(region_rename_table['chr']))
    if path is None:
        chroms_str = _get_chrom_str(chrom_ids)
        if saved_root is None:
            raise Exception()
        path = os.path.join(saved_root,'chrom_{}.tsv'.format(chroms_str))
    data.to_csv(path,header=False,index=None)

def _get_min_chrom(fai):
    min_chrom = None
    min_length = None
    for chrom,length in fai.items():
        if min_chrom is None:
            min_chrom = str(chrom)
            min_length = length
        elif min_length > length:
            min_chrom = str(chrom)
            min_length = length
    return min_chrom
    
def split_by_chr(region_rename_table,fai,saved_root,split_with_strand=False):
    test_chrom = _get_min_chrom(fai)
    region_rename_table['length'] = region_rename_table['end'] - region_rename_table['start'] + 1
    train_val_chrom_ids = [str(chrom) for chrom in fai.keys() if chrom != test_chrom]
    
    #Write threshold
    train_val_length = list(region_rename_table[region_rename_table['chr'].isin(train_val_chrom_ids)]['length'])
    path = os.path.join(saved_root,'threshold.txt')
    with open(path,'w') as fp:
        fp.write(str(int(np.median(train_val_length)*2)))
    
    #Write dataset table
    train_val_chrom_ids.sort()
    if split_with_strand:
        for val_chrom_ in train_val_chrom_ids:
            train_chroms_ = list(train_val_chrom_ids)
            train_chroms_.remove(val_chrom_)
            train_table_ = _get_subtable(region_rename_table,train_chroms_)
            val_table_ = _get_subtable(region_rename_table,val_chrom_)
            for strand in ['+','-']:
                val_table = val_table_[val_table_['strand'] == strand]
                train_table = train_table_.append(val_table_[val_table_['strand'] != strand])
                if strand == '+':
                    train_chroms = train_chroms_ + ['{}_minus'.format(val_chrom_)]
                    val_chrom = '{}_plus'.format(val_chrom_)
                else:
                    train_chroms = train_chroms_ + ['{}_plus'.format(val_chrom_)]
                    val_chrom = '{}_minus'.format(val_chrom_)
                train_chrom_str = _get_chrom_str(train_chroms)
                val_chrom_str = _get_chrom_str(val_chrom)
                train_path = os.path.join(saved_root,'chrom_{}.tsv'.format(train_chrom_str))
                val_path = os.path.join(saved_root,'chrom_{}.tsv'.format(val_chrom_str))
                write_id_table(train_table,path=train_path)
                write_id_table(val_table,path=val_path)

    else:
        for val_chrom in train_val_chrom_ids:
            train_chroms = list(train_val_chrom_ids)
            train_chroms.remove(val_chrom)
            train_table = _get_subtable(region_rename_table,train_chroms)
            val_table = _get_subtable(region_rename_table,val_chrom)
            write_id_table(train_table,saved_root=saved_root)
            write_id_table(val_table,saved_root=saved_root)

    #Write training and validation table
    train_val_table = _get_subtable(region_rename_table,train_val_chrom_ids)
    write_id_table(train_val_table,saved_root=saved_root)
    #Write test table
    test_path = os.path.join(saved_root,'test_chrom_{}.tsv'.format(test_chrom))
    test_table = _get_subtable(region_rename_table,test_chrom)
    write_id_table(test_table,path=test_path)
    
def _flatten_chunck_list(chunck_list):
    flatten_list = []
    for list_ in chunck_list:
        flatten_list += list_
    return flatten_list
    
def split_by_num(region_rename_table,fai,saved_root,fold_num,split_with_strand=False):
    fold_num += 1
    region_rename_table['length'] = region_rename_table['end'] - region_rename_table['start'] + 1
    chroms = sorted([str(chrom) for chrom in fai.keys()])
    chrom_num = len(chroms)
    chroms_list = []
    usage = {}
    
    step = int(chrom_num/fold_num)
    if chrom_num%fold_num != 0:
        step += 1
    
    for i in range(fold_num):
        chrom_chunck = chroms[i*step:(i+1)*step]
        usage["dataset_{}".format(i+1)] = chrom_chunck
        chroms_list.append(chrom_chunck)

    write_json(usage,os.path.join(saved_root,'data_split.json'))
    train_val_chrom_list = chroms_list[:-1]
    test_chrom_list = chroms_list[-1]
    train_val_chrom_ids = _flatten_chunck_list(train_val_chrom_list)

    #Write threshold
    train_val_length = list(region_rename_table[region_rename_table['chr'].isin(train_val_chrom_ids)]['length'])
    path = os.path.join(saved_root,'threshold.txt')
    with open(path,'w') as fp:
        fp.write(str(int(np.median(train_val_length)*2)))
    
    #Write dataset table
    if split_with_strand:
        raise Exception("Currently split_with_strand is not supported with fold_num")
    else:
        for i in range(len(train_val_chrom_list)):
            train_chrom_list = train_val_chrom_list[i+1:] + train_val_chrom_list[:i]
            train_chroms = _flatten_chunck_list(train_chrom_list)
            val_chroms = train_val_chrom_list[i]
            train_path = os.path.join(saved_root,'train_dataset_{}.tsv'.format(i+1))
            val_path = os.path.join(saved_root,'val_dataset_{}.tsv'.format(i+1))
            train_table = _get_subtable(region_rename_table,train_chroms)
            val_table = _get_subtable(region_rename_table,val_chroms)
            write_id_table(train_table,path=train_path)
            write_id_table(val_table,path=val_path)

    #Write training and validation table
    train_val_table_path = os.path.join(saved_root,'train_val_dataset.tsv')
    train_val_table = _get_subtable(region_rename_table,train_val_chrom_ids)
    write_id_table(train_val_table,path=train_val_table_path)
    #Write test table
    test_path = os.path.join(saved_root,'test_dataset.tsv')
    test_table = _get_subtable(region_rename_table,test_chrom_list)
    write_id_table(test_table,path=test_path)

def main(fai_path,region_rename_table_path,saved_root,split_with_strand=False,fold_num=None):
    fai = read_fai(fai_path)
    region_rename_table = pd.read_csv(region_rename_table_path,sep='\t',dtype={'chr':str,'start':int,'end':int})                    
    if fold_num is None:
        split_by_chr(region_rename_table,fai,saved_root,split_with_strand)
    else:
        split_by_num(region_rename_table,fai,saved_root,fold_num,split_with_strand)
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program split data")
    parser.add_argument("--region_rename_table_path",required=True)
    parser.add_argument("--fai_path",required=True)
    parser.add_argument("--saved_root",required=True)
    parser.add_argument("--fold_num",type=int,default=None,help='The number of training and validation datasets')
    parser.add_argument("--split_with_strand",action='store_true',help='Splitting trainnig and validation data with strand')
    args = parser.parse_args()
                        
    main(args.fai_path,args.region_rename_table_path,args.saved_root,args.split_with_strand,args.fold_num)
