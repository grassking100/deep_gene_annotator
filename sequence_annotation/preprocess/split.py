import os,sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_fai, write_json, read_region_table
from sequence_annotation.utils.stats import mad_threshold
    
def _flatten_chunck_list(chunck_list):
    flatten_list = []
    for list_ in chunck_list:
        flatten_list += list_
    return flatten_list

def _get_subtable(table,chrom_ids):
    if not isinstance(chrom_ids,list):
        chrom_ids = [chrom_ids]
    return table[table['chr'].isin(chrom_ids)]

def _get_chrom_str(chrom_ids):
    if not isinstance(chrom_ids,list):
        chrom_ids = [chrom_ids]
    chrom_ids = sorted(list(set(chrom_ids)))
    chroms_str = '_'.join([str(chrom) for chrom in sorted(chrom_ids)])
    return chroms_str

def _get_region_id_path(saved_root,name):
    path = os.path.join(saved_root,'{}.txt'.format(name))
    return path

def _export_region_data(region_rename_table,saved_root,name):
    region_id_path = _get_region_id_path(saved_root,name)
    region_ids = region_rename_table['new_id'].drop_duplicates()
    region_ids.to_csv(region_id_path,header=False,index=None)
    return region_id_path

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

def _write_stats_data(lengths,path):
    stats = {}
    stats['max'] = max(lengths)
    stats['min'] = min(lengths)
    stats['median'] = np.median(lengths)
    stats['mean'] = np.mean(lengths)
    stats['MAD derived threshold'] = mad_threshold(lengths)
    write_json(stats,path)

def get_relative_paths(paths):
    relative_paths = []
    for path_dict in paths:
        relative_path = {}
        for key,path in path_dict.items():
            relative_path[key] = path.split('/')[-1]
        relative_paths.append(relative_path)
    return relative_paths
    
def split_by_chr(region_rename_table,fai,saved_root,split_with_strand=False):
    test_chrom = _get_min_chrom(fai)
    train_val_chrom_ids = [str(chrom) for chrom in fai.keys() if chrom != test_chrom]
    #Write dataset table
    train_val_chrom_ids.sort()
    split_table = []
    train_chrom_list = []
    val_chrom_list = []
    train_tables = []
    val_tables = []
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
                    
                train_chrom_list.append(train_chroms)
                val_chrom_list.append(val_chrom)
                train_tables.append(train_table)
                val_tables.append(val_table)

    else:
        for val_chrom in train_val_chrom_ids:
            train_chroms = list(train_val_chrom_ids)
            train_chroms.remove(val_chrom)
            train_table = _get_subtable(region_rename_table,train_chroms)
            val_table = _get_subtable(region_rename_table,val_chrom)
            
            train_chrom_list.append(train_chroms)
            val_chrom_list.append(val_chrom)
            train_tables.append(train_table)
            val_tables.append(val_table)

    for train_table,val_table,train_chroms,val_chrom in zip(train_tables,val_tables,
                                                            train_chrom_list,val_chrom_list):
        
        train_name = 'train_{}'.format(_get_chrom_str(train_chroms))
        val_name = 'val_{}'.format(_get_chrom_str(val_chrom))
        train_id_path = _export_region_data(train_table,saved_root,train_name)
        val_id_path = _export_region_data(val_table,saved_root,val_name)
        split_table.append({'training_path':train_id_path,'validation_path':val_id_path})

    #Write training and validation table
    train_val_table = _get_subtable(region_rename_table,train_val_chrom_ids)
    train_val_name = 'train_val_{}'.format(_get_chrom_str(train_val_chrom_ids))
    _export_region_data(train_val_table,saved_root,train_val_name)

    #Write statistic result
    train_val_length = list(region_rename_table[region_rename_table['chr'].isin(train_val_chrom_ids)]['length'])
    path = os.path.join(saved_root,'train_val_stats.json')
    _write_stats_data(train_val_length,path)

    #Write test table
    test_table = _get_subtable(region_rename_table,test_chrom)
    test_name = 'test_{}'.format(_get_chrom_str(test_chrom))
    test_id_path = _export_region_data(test_table,saved_root,test_name)

    for item in split_table:
        item['testing_path'] = test_id_path
        
    split_table = get_relative_paths(split_table)
    split_table = pd.DataFrame.from_dict(split_table)[['training_path','validation_path','testing_path']]
    split_table_path = os.path.join(saved_root,'split_table.csv')
    split_table.to_csv(split_table_path,index=None)

def split_by_num(region_rename_table,fai,saved_root,fold_num,split_with_strand=False):
    region_rename_table['length'] = region_rename_table['end'] - region_rename_table['start'] + 1
    chroms = OrderedDict(sorted(fai.items(), key=lambda x: (x[1],x[0]),reverse=True))
    
    split_status = {}
    chroms_list = []
    for _ in range(fold_num+1):
        chroms_list.append([])

    for index,chrom in enumerate(chroms):
        index = index%(fold_num+1)
        chroms_list[index].append(chrom)
    
    for index,item in enumerate(chroms_list):
        split_status["dataset_{}".format(index+1)] = item    

    write_json(split_status,os.path.join(saved_root,'data_split.json'))
    train_val_chrom_list = chroms_list[:-1]
    test_chrom_list = chroms_list[-1]
    train_val_chrom_ids = _flatten_chunck_list(train_val_chrom_list)

    split_table = []
    #Write dataset table
    if split_with_strand:
        raise Exception("Currently split_with_strand is not supported with fold_num")
    else:
        for i in range(len(train_val_chrom_list)):
            train_chrom_list = train_val_chrom_list[i+1:] + train_val_chrom_list[:i]
            train_chroms = _flatten_chunck_list(train_chrom_list)
            val_chroms = train_val_chrom_list[i]
            train_table = _get_subtable(region_rename_table,train_chroms)
            val_table = _get_subtable(region_rename_table,val_chroms)
            train_name = os.path.join(saved_root,'train_dataset_{}'.format(i+1))
            val_name = os.path.join(saved_root,'val_dataset_{}'.format(i+1))
            train_id_path = _export_region_data(train_table,saved_root,train_name)
            val_id_path = _export_region_data(val_table,saved_root,val_name)
            split_table.append({'training_path':train_id_path,'validation_path':val_id_path})

    #Write training and validation table
    train_val_table = _get_subtable(region_rename_table,train_val_chrom_ids)
    _export_region_data(train_val_table,saved_root,'train_val_dataset')
    
    #Write statistic result
    train_val_length = list(region_rename_table[region_rename_table['chr'].isin(train_val_chrom_ids)]['length'])
    path = os.path.join(saved_root,'train_val_stats.json')
    _write_stats_data(train_val_length,path)
    
    #Write test table
    test_table = _get_subtable(region_rename_table,test_chrom_list)
    test_id_path = _export_region_data(test_path,saved_root,'test_dataset')

    for item in split_table:
        item['testing_path'] = test_id_path
        
    split_table = get_relative_paths(split_table)
    split_table = pd.DataFrame.from_dict(split_table)[['training_path','validation_path','testing_path']]
    split_table_path = os.path.join(saved_root,'split_table.csv')
    split_table.to_csv(split_table_path,index=None)
    
def main(fai_path,region_rename_table_path,saved_root,split_with_strand=False,fold_num=None):
    fai = read_fai(fai_path)
    region_rename_table = read_region_table(region_rename_table_path)
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
