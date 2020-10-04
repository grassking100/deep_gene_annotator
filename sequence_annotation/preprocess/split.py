import os,sys
import pandas as pd
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json, create_folder
from sequence_annotation.file_process.utils import read_fai
from sequence_annotation.file_process.get_region_table import read_region_table,get_region_table_from_fai,write_region_table

def _merge_strand_chrom(chrom_ids):
    id_list = []
    chrom_strands = {}
    for id_ in chrom_ids:
        chrom_id,strand = id_
        if chrom_id not in chrom_strands:
            chrom_strands[chrom_id] = []
        chrom_strands[chrom_id].append(strand)
    
    for chrom_id,strands in chrom_strands.items():
        if len(strands)>2:
            raise Exception("Wrong number of strand in {}".format(chrom_id))
        if len(strands)==2:
            id_list.append(chrom_id)
        else:
            if strands[0] == '+':
                id_list.append("{}_plus".format(chrom_id))
            elif strands[0] == '-':
                id_list.append("{}_minus".format(chrom_id))
            else:
                raise Exception()
    return id_list


def _get_chrom_name(chrom_ids,treat_strand_independent=False):
    if treat_strand_independent:
        chrom_ids = _merge_strand_chrom(chrom_ids)
    name = '_'.join([chrom for chrom in sorted(chrom_ids)])
    return name

    
def _get_min_chrom(fai):
    min_chrom = None
    min_length = None
    for chrom,length in fai.items():
        if min_chrom is None or min_length > length:
            min_chrom = chrom
            min_length = length
    return min_chrom


def grouping_by_fold(fai,fold_num):
    groups = {}
    lengths = {}
    names = []
    for index in range(1,1+fold_num):
        name = "dataset_{}".format(index)
        names.append(name)
        groups[name] = []
        lengths[name] = 0
    
    for chrom,length in sorted(fai.items(), key=lambda x: (x[1],x[0]),reverse=True):
        min_dataset_id = sorted(lengths.items(), key=lambda x: (x[1],x[0]))[0][0]
        groups[min_dataset_id].append(chrom)
        lengths[min_dataset_id] += length
    train_val_group = {}
    for name in names[:-1]:
        train_val_group[name] = groups[name]
    test_name = names[-1]
    test_ids = groups[test_name]
    return train_val_group,test_name,test_ids


def grouping_by_chrom(fai):
    chroms = list(fai.keys())
    train_val_group = {}
    test_chrom = _get_min_chrom(fai)
    test_ids = [test_chrom]
    for chrom in chroms:
        if chrom != test_chrom:
            train_val_group[chrom] = [chrom]
    return train_val_group,test_chrom,test_ids
    

def split_by_chrom_and_strand(train_val_group,test_ids,region_table,treat_strand_independent=False):
    test_table = region_table[region_table['chr'].isin(test_ids)].copy()
    train_val_tables = {}
    for dataset_name,chroms in train_val_group.items():
        for chrom in chroms:
            if treat_strand_independent:
                for strand in ['+','-']:
                    subtable = region_table[(region_table['chr']==chrom) & (region_table['strand']==strand)]
                    train_val_tables[(dataset_name,strand)] = subtable.copy()
            else:
                subtable = region_table[region_table['chr']==chrom]
                train_val_tables[dataset_name] = subtable.copy()
    return train_val_tables,test_table


def split(fai_path,output_root,region_table_path=None,
                 treat_strand_independent=False,fold_num=None):
    create_folder(output_root)
    fai = read_fai(fai_path)
    if region_table_path is not None:
        region_table = read_region_table(region_table_path)
    else:
        region_table = get_region_table_from_fai(fi)
    #Split by chromosome and strand
    if fold_num is not None:
        train_val_group,test_name,test_chroms = grouping_by_fold(fai,fold_num)
    else:
        train_val_group,test_name,test_chroms = grouping_by_chrom(fai)
    write_json(train_val_group,os.path.join(output_root,"train_val_groups.json"))
    write_json({test_name:test_chroms},os.path.join(output_root,"test_groups.json"))
    result = split_by_chrom_and_strand(train_val_group,test_chroms,region_table,treat_strand_independent)
    train_val_tables,test_table = result
    name_list = []
    names = {'cross-validation':[]}
    #Write training and validation table
    train_val_chroms = list(train_val_tables.keys())
    train_val_name = "train_val_"+_get_chrom_name(train_val_chroms,treat_strand_independent)
    train_val_path = '{}_region_table.tsv'.format(train_val_name)
    train_val_table = pd.concat(list(train_val_tables.values()))
    write_region_table(train_val_table,os.path.join(output_root,train_val_path))
    names['training and validation'] = train_val_name
    name_list.append(train_val_name)
    #Write test table
    test_name = "test_"+_get_chrom_name([test_name])
    test_path = '{}_region_table.tsv'.format(test_name)
    write_region_table(test_table,os.path.join(output_root,test_path))
    names['testing'] = test_name
    name_list.append(test_name)
    #Export ID list by their belonging in each group
    if len(train_val_tables)>=2:
        for val_chrom,val_table in train_val_tables.items():
            train_table = [table for key,table in train_val_tables.items() if key!=val_chrom]
            train_chroms = [key for key in train_val_tables.keys() if key!=val_chrom]
            train_table = pd.concat(train_table)
            train_name = "train_"+_get_chrom_name(train_chroms,treat_strand_independent)
            val_name = "val_"+_get_chrom_name([val_chrom],treat_strand_independent)
            name_list.append(train_name)
            name_list.append(val_name)
            train_path = '{}_region_table.tsv'.format(train_name)
            val_path = '{}_region_table.tsv'.format(val_name)
            write_region_table(train_table,os.path.join(output_root,train_path))
            write_region_table(val_table,os.path.join(output_root,val_path))
            names['cross-validation'].append({'training':train_name,
                                              'validation':val_name})
    name_path = os.path.join(output_root,'name.json')
    name_list_path = os.path.join(output_root,'name_list.json')
    write_json(names,name_path)
    write_json(name_list,name_list_path)
    
