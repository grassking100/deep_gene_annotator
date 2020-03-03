import os,sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_fai, write_json, read_region_table

def _get_chrom_str(chrom_ids):
    if not isinstance(chrom_ids,list):
        chrom_ids = [chrom_ids]
    old_ids = chrom_ids
    chrom_ids = []

    for id_ in old_ids:
        split_data = id_.split('_')
        if len(split_data)>=2:
            if split_data[-1] == '+':
                chrom_ids.append('_'.join(split_data[:-1])+"_plus")
            elif split_data[-1] == '-':
                chrom_ids.append('_'.join(split_data[:-1])+"_minus")
            else:
                chrom_ids.append(id_)
        else:
            chrom_ids.append(id_)
    
    chroms = []
    for id_ in chrom_ids:
        chroms.append('_'.join(id_.split('_')[:-1]))
        
    for chrom in chroms:
        plus_id = "{}_plus".format(chrom)
        minus_id = "{}_minus".format(chrom)
        if plus_id in chrom_ids and minus_id in chrom_ids:
            chrom_ids.remove(plus_id)
            chrom_ids.remove(minus_id)
            chrom_ids.append(chrom)

    chrom_ids = sorted(list(set(chrom_ids)))
    chroms_str = '_'.join([str(chrom) for chrom in sorted(chrom_ids)])
    return chroms_str

def _export_region_data(region_rename_table,region_id_path,id_source):
    region_ids = region_rename_table[id_source].drop_duplicates()
    region_ids.to_csv(region_id_path,header=False,index=None)

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
    write_json(stats,path)

def get_relative_paths(paths):
    relative_paths = []
    for path_dict in paths:
        relative_path = {}
        for key,path in path_dict.items():
            relative_path[key] = path.split('/')[-1]
        relative_paths.append(relative_path)
    return relative_paths

def grouping(fai,fold_num):
    chroms = OrderedDict(sorted(fai.items(), key=lambda x: (x[1],x[0]),reverse=True))
    chroms_list = [[] for _ in range(fold_num+1)]
    for index,chrom in enumerate(chroms):
        index = index%(fold_num+1)
        chroms_list[index].append(chrom)
    train_val_group = {}
    test_group = {}
    for index,item in enumerate(chroms_list[:-1]):
        train_val_group["dataset_{}".format(index+1)] = item
    test_group["dataset_{}".format(fold_num+1)] = chroms_list[-1]
    return train_val_group,test_group

def main(fai_path,region_path,id_source,saved_root,treat_strand_independent=False,fold_num=None):
    fai = read_fai(fai_path)
    region_rename_table = read_region_table(region_path)
    region_rename_table['chr_strand'] = region_rename_table['chr'] + "_" + region_rename_table['strand']
    #Split by chromosome and strand
    if fold_num is None:
        #Splitting Training and valdation chromosomes and testing chromosome
        test_chrom_id = _get_min_chrom(fai)
        
        test_table = region_rename_table[region_rename_table['chr']==test_chrom_id]
        train_val_table = region_rename_table[region_rename_table['chr']!=test_chrom_id]
        #Assign belonging in train_val_table
        if treat_strand_independent:
            old_ids = [chrom for chrom in fai.keys() if chrom != test_chrom_id]
            train_val_chrom_ids = ['{}_+'.format(id_) for id_ in  old_ids]
            train_val_chrom_ids += ['{}_-'.format(id_) for id_ in  old_ids]
            train_val_chrom_ids = sorted(train_val_chrom_ids)
            train_val_table = train_val_table.assign(belonging=train_val_table['chr_strand'])
        else:
            train_val_chrom_ids = sorted([chrom for chrom in fai.keys() if chrom != test_chrom_id])
            train_val_table = train_val_table.assign(belonging=train_val_table['chr'])   
    else:
        train_val_group,test_group = grouping(fai,fold_num)
        write_json(train_val_group,os.path.join(saved_root,'train_val_group.json'))
        write_json(test_group,os.path.join(saved_root,'test_group.json'))
        if treat_strand_independent:
            group = dict(test_group)
            old_group = train_val_group
            train_val_group = {}
            train_val_chrom_ids = []
            for id_,list_ in old_group.items():
                train_val_group[id_+"_+"] = ["{}_+".format(item) for item in list_]
                train_val_group[id_+"_-"] = ["{}_-".format(item) for item in list_]
                train_val_chrom_ids += [id_+"_+",id_+"_-"]
            group = dict(train_val_group)
            train_val_chrom_ids = sorted(train_val_chrom_ids)
        else:
            group = dict(train_val_group)
            group.update(test_group)
            train_val_chrom_ids = sorted(list(train_val_group.keys()))

        test_chrom_id = list(test_group.keys())[0]
        test_origin_ids = list(test_group.values())[0]
        test_table = region_rename_table[region_rename_table['chr'].isin(test_origin_ids)]
        train_val_table = region_rename_table[~region_rename_table['chr'].isin(test_origin_ids)]
        #Assign belonging in train_val_table
        train_val_table = train_val_table.assign(belonging=None)
        for id_ in train_val_chrom_ids:
            origin_ids = group[id_]
            if treat_strand_independent:
                train_val_table.loc[train_val_table['chr_strand'].isin(origin_ids),'belonging'] = id_
            else:
                train_val_table.loc[train_val_table['chr'].isin(origin_ids),'belonging'] = id_
                
    #Write test table
    test_path = os.path.join(saved_root,'test_{}.txt'.format(_get_chrom_str(test_chrom_id)))
    _export_region_data(test_table,test_path,id_source)
        
    split_table = []
    #Export ID list by their belonging in each group
    if len(train_val_chrom_ids)>=2:
        for index,val_chrom in enumerate(train_val_chrom_ids):
            train_chroms = list(train_val_chrom_ids)
            train_chroms.remove(val_chrom)

            train_table = train_val_table[train_val_table['belonging'].isin(train_chroms)]
            val_table = train_val_table[train_val_table['belonging']==val_chrom]
            train_path = os.path.join(saved_root,'train_{}.txt'.format(_get_chrom_str(train_chroms)))
            val_path = os.path.join(saved_root,'val_{}.txt'.format(_get_chrom_str(val_chrom)))
            _export_region_data(train_table,train_path,id_source)
            _export_region_data(val_table,val_path,id_source)
            split_table.append({'training_path':train_path,'validation_path':val_path,
                                'testing_path':test_path})

    #Write training and validation table
    train_val_path = os.path.join(saved_root,'train_val_{}.txt'.format(_get_chrom_str(train_val_chrom_ids)))
    _export_region_data(train_val_table,train_val_path,id_source)
    
    #Write statistic result
    train_val_length = list(train_val_table['length'])
    path = os.path.join(saved_root,'train_val_stats.json')
    _write_stats_data(train_val_length,path)

    if len(split_table) >= 1:
        split_table = get_relative_paths(split_table)
        split_table = pd.DataFrame.from_dict(split_table)[['training_path','validation_path','testing_path']]
        split_table_path = os.path.join(saved_root,'split_table.csv')
        split_table.to_csv(split_table_path,index=None)
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program export texts which have region ids be split")
    parser.add_argument("--region_path",help='The path of region_rename_table_both_strand.tsv',required=True)
    parser.add_argument("--id_source",help='Id source in region_rename_table_both_strand.tsv',required=True)
    parser.add_argument("--fai_path",required=True)
    parser.add_argument("--saved_root",required=True)
    parser.add_argument("--fold_num",type=int,default=None,help='If it is None, then dataset would be split by chromosome and strand, otherwise'\
                       ' it would be split to specific number of datasets.')
    parser.add_argument("--treat_strand_independent",action='store_true',help="Each strand on training and validation dataset"\
                        "would be treat independently")
    args = parser.parse_args()
                        
    main(args.fai_path,args.region_path,args.id_source,
         args.saved_root,args.treat_strand_independent,
         args.fold_num)
