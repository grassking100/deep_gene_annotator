import os,sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_bed,read_fai,write_json

def write_id_table(region_rename_table,chrom_ids,saved_root=None,path=None):
    data = region_rename_table[region_rename_table['chr'].isin(chrom_ids)]['new_id'].drop_duplicates()
    if path is None:
        if len(chrom_ids)==1:
            chroms_str = str(chrom_ids)[0]
        else:
            chroms_str = '_'.join([str(chrom) for chrom in chrom_ids])
        if saved_root is None:
            raise Exception()
        path = os.path.join(saved_root,'chrom_{}_new_id.tsv'.format(chroms_str))
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
    
def split_by_chr(region_rename_table,fai,saved_root):
    test_chrom = _get_min_chrom(fai)
    region_rename_table['length'] = region_rename_table['end'] - region_rename_table['start'] + 1
    train_val_chrom_ids = [str(chrom) for chrom in fai.keys() if chrom != test_chrom]
    train_val_length = list(region_rename_table[region_rename_table['chr'].isin(train_val_chrom_ids)]['length'])
    path = os.path.join(saved_root,'threshold.txt')
    with open(path,'w') as fp:
        fp.write(str(int(np.median(train_val_length)*2)))
    
    train_val_chrom_ids.sort()
    for val_chrom in train_val_chrom_ids:
        train_chroms = list(train_val_chrom_ids)
        train_chroms.remove(val_chrom)
        write_id_table(region_rename_table,train_chroms,saved_root)
        write_id_table(region_rename_table,[val_chrom],saved_root)

    write_id_table(region_rename_table,train_val_chrom_ids,saved_root)
    test_path = os.path.join(saved_root,'test_chrom_{}_new_id.tsv'.format(test_chrom))
    write_id_table(region_rename_table,[test_chrom],path=test_path)
    
def _flatten_chunck_list(chunck_list):
    flatten_list = []
    for list_ in chunck_list:
        flatten_list += list_
    return flatten_list
    
def split_by_num(region_rename_table,fai,saved_root,fold_num):
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

    train_val_length = list(region_rename_table[region_rename_table['chr'].isin(train_val_chrom_ids)]['length'])
    path = os.path.join(saved_root,'threshold.txt')
    with open(path,'w') as fp:
        fp.write(str(int(np.median(train_val_length)*2)))
    
    for i in range(len(train_val_chrom_list)):
        train_chrom_list = train_val_chrom_list[i+1:] + train_val_chrom_list[:i]
        train_chroms = _flatten_chunck_list(train_chrom_list)
        val_chroms = train_val_chrom_list[i]
        train_table_path = os.path.join(saved_root,'train_dataset_{}_new_id.tsv'.format(i+1))
        val_table_path = os.path.join(saved_root,'val_dataset_{}_new_id.tsv'.format(i+1))
        write_id_table(region_rename_table,train_chroms,path=train_table_path)
        write_id_table(region_rename_table,val_chroms,path=val_table_path)

    train_val_table_path = os.path.join(saved_root,'train_val_dataset_new_id.tsv')
    write_id_table(region_rename_table,train_val_chrom_ids,path=train_val_table_path)
    test_path = os.path.join(saved_root,'test_new_id.tsv')
    write_id_table(region_rename_table,test_chrom_list,path=test_path)

def main(fai_path,region_rename_table_path,saved_root,fold_num=None):
    fai = read_fai(fai_path)
    region_rename_table = pd.read_csv(region_rename_table_path,sep='\t',dtype={'chr':str,'start':int,'end':int})                    
    if fold_num is None:
        split_by_chr(region_rename_table,fai,saved_root)
    else:
        split_by_num(region_rename_table,fai,saved_root,fold_num)
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program split data")
    parser.add_argument("--region_rename_table_path",required=True)
    parser.add_argument("--fai_path",required=True)
    parser.add_argument("--saved_root",required=True)
    parser.add_argument("--fold_num",type=int,default=None)
    args = parser.parse_args()
                        
    main(args.fai_path,args.region_rename_table_path,args.saved_root,args.fold_num)
