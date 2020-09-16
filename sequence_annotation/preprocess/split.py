import os,sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json, create_folder
from sequence_annotation.file_process.utils import read_fai,read_bed,write_gff
from sequence_annotation.file_process.get_region_table import read_region_table
from sequence_annotation.file_process.get_id_table import get_id_convert_dict
from sequence_annotation.file_process.get_subbed import get_subbed
from sequence_annotation.file_process.bed2gff import bed2gff
from sequence_annotation.file_process.get_subfasta import main as get_subfasta_main
from sequence_annotation.file_process.get_sub_region_table import main as get_sub_region_table_main
from sequence_annotation.file_process.gff_analysis import main as gff_analysis_main

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


def _export_region_data(region_table,region_id_path,id_source):
    region_ids = region_table[id_source].drop_duplicates()
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


def split_by_chrom_and_strand(fai_path,region_table_path,treat_strand_independent=False):
    fai = read_fai(fai_path)
    region_table = read_region_table(region_table_path)
    region_table['chr_strand'] = region_table['chr'] + "_" + region_table['strand']
    #Splitting Training and valdation chromosomes and testing chromosome
    test_chrom_id = _get_min_chrom(fai)
    test_table = region_table[region_table['chr']==test_chrom_id]
    train_val_table = region_table[region_table['chr']!=test_chrom_id]
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
    return train_val_chrom_ids,train_val_table,test_chrom_id,test_table


def split_by_fold(fai_path,region_table_path,fold_num=None):
    train_val_group,test_group = grouping(fai,fold_num)
    lengths = {}
    for group_id,group in train_val_group.items():
        lengths[group_id] = sum([fai[id_] for id_ in group])
    for group_id,group in test_group.items():
        lengths[group_id] = sum([fai[id_] for id_ in group])
    write_json(lengths,os.path.join(output_root,'lengths.json'))
    write_json(train_val_group,os.path.join(output_root,'train_val_group.json'))
    write_json(test_group,os.path.join(output_root,'test_group.json'))
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
    test_table = region_table[region_table['chr'].isin(test_origin_ids)]
    train_val_table = region_table[~region_table['chr'].isin(test_origin_ids)]
    #Assign belonging in train_val_table
    train_val_table = train_val_table.assign(belonging=None)
    for id_ in train_val_chrom_ids:
        origin_ids = group[id_]
        if treat_strand_independent:
            train_val_table.loc[train_val_table['chr_strand'].isin(origin_ids),'belonging'] = id_
        else:
            train_val_table.loc[train_val_table['chr'].isin(origin_ids),'belonging'] = id_
    return train_val_chrom_ids,train_val_table,test_chrom_id,test_table


def split_region(fai_path,region_table_path,id_source,output_root,treat_strand_independent=False,fold_num=None):
    create_folder(output_root)
    #Split by chromosome and strand
    if fold_num is None:
        result = split_by_chrom_and_strand(fai_path,region_table_path,treat_strand_independent)
    else:
        result = split_by_fold(fai_path,region_table_path,fold_num)
    train_val_chrom_ids,train_val_table,test_chrom_id,test_table = result
    #Write test table
    test_path = os.path.join(output_root,'test_{}.txt'.format(_get_chrom_str(test_chrom_id)))
    _export_region_data(test_table,test_path,id_source)
    split_table = []
    #Export ID list by their belonging in each group
    if len(train_val_chrom_ids)>=2:
        for index,val_chrom in enumerate(train_val_chrom_ids):
            train_chroms = list(train_val_chrom_ids)
            train_chroms.remove(val_chrom)
            train_table = train_val_table[train_val_table['belonging'].isin(train_chroms)]
            val_table = train_val_table[train_val_table['belonging']==val_chrom]
            train_path = os.path.join(output_root,'train_{}.txt'.format(_get_chrom_str(train_chroms)))
            val_path = os.path.join(output_root,'val_{}.txt'.format(_get_chrom_str(val_chrom)))
            _export_region_data(train_table,train_path,id_source)
            _export_region_data(val_table,val_path,id_source)
            split_table.append({'training_path':train_path,'validation_path':val_path,
                                'testing_path':test_path})
    #Write training and validation table
    train_val_path = os.path.join(output_root,'train_val_{}.txt'.format(_get_chrom_str(train_val_chrom_ids)))
    _export_region_data(train_val_table,train_val_path,id_source)
    train_val_test_paths = {'train_val_path':train_val_path,'test_path':test_path}
    write_json(train_val_test_paths,os.path.join(output_root,'train_val_test_path.json'))
    #Write statistic result
    train_val_length = list(train_val_table['length'])
    path = os.path.join(output_root,'train_val_stats.json')
    _write_stats_data(train_val_length,path)
    if len(split_table) >= 1:
        split_table = get_relative_paths(split_table)
        split_table = pd.DataFrame.from_dict(split_table)[['training_path','validation_path','testing_path']]
        split_table_path = os.path.join(output_root,'split_table.csv')
        split_table.to_csv(split_table_path,index=None)
    
    
def main(fai_path,id_convert_table_path,processed_root,output_root,fold_num=None,
         treat_strand_independent=False,on_double_strand_data=False):
    result_root = os.path.join(processed_root,'result')
    region_table_path = os.path.join(result_root,'region_table.tsv')
    ds_root = os.path.join(result_root,'double_strand')
    ss_root = os.path.join(result_root,'single_strand')
    ss_canonical_id_convert_path=os.path.join(ss_root,"ss_canonical_id_convert.tsv")
    region_num_path=os.path.join(output_root,"region_num.json")
    fasta_root=os.path.join(output_root,"fasta")
    gff_root=os.path.join(output_root,"gff")
    id_root=os.path.join(output_root,"id")
    region_table_root=os.path.join(output_root,"region_table")
    id_convert_dict = get_id_convert_dict(id_convert_table_path)
    ss_canonical_id_convert_dict = get_id_convert_dict(ss_canonical_id_convert_path)
    create_folder(output_root)
    create_folder(fasta_root)
    create_folder(gff_root)
    create_folder(region_table_root)
    
    if treat_strand_independent and not on_double_strand_data:
        treat_strand_independent = True
        
    if on_double_strand_data:
        id_source="ordinal_id_wo_strand"
        output_region_fasta_root=os.path.join(ds_root,'ds_region.fasta')
        rna_bed_path=os.path.join(ds_root,"ds_rna.bed")
        canonical_bed_path=os.path.join(ds_root,"ds_canonical.bed")
    else:
        id_source="ordinal_id_with_strand"
        output_region_fasta_root=os.path.join(ss_root,'ss_region.fasta')
        rna_bed_path=os.path.join(ss_root,"ss_rna.bed")
        canonical_bed_path=os.path.join(ss_root,"ss_canonical.bed")
        
    split_region(fai_path,region_table_path,id_source,id_root,fold_num=fold_num,
                 treat_strand_independent=treat_strand_independent)
    
    region_num = {}
    rna_bed = read_bed(rna_bed_path)
    canonical_bed = read_bed(canonical_bed_path)
    for name in os.listdir(id_root):
        name = name.split('.')
        if name[-1] == 'txt':
            file_name = '.'.join(name[:-1])
            region_id_path=os.path.join(id_root,"{}.txt".format(file_name))
            fasta_path=os.path.join(fasta_root,"{}.fasta".format(file_name))
            part_region_table_path=os.path.join(region_table_root,"{}_part_region_table.tsv".format(file_name))
            if on_double_strand_data:
                part_gff_path=os.path.join(gff_root,"ds_{}.gff3".format(file_name))
                part_canonical_gff_path=os.path.join(gff_root,"ds_{}_canonical.gff3".format(file_name))
                canonical_stats_root=os.path.join(output_root,"ds_canonical_stats","{}".format(file_name))
                rna_stats_root=os.path.join(output_root,"ds_rna_stats","{}".format(file_name))
            else:
                part_gff_path=os.path.join(gff_root,"ss_{}.gff3".format(file_name))
                part_canonical_gff_path=os.path.join(gff_root,"ss_{}_canonical.gff3".format(file_name))
                canonical_stats_root=os.path.join(output_root,"ss_canonical_stats","{}".format(file_name))
                rna_stats_root=os.path.join(output_root,"ss_rna_stats","{}".format(file_name))
            region_ids = list(pd.read_csv(region_id_path,header=None,sep='\t')[0])
            region_num[file_name] = len(region_ids)
            part_rna_bed = get_subbed(rna_bed,region_ids,query_column='chr')
            part_canonical_bed = get_subbed(canonical_bed,region_ids,query_column='chr')
            write_gff(bed2gff(part_rna_bed,id_convert_dict),part_gff_path)
            write_gff(bed2gff(part_canonical_bed,ss_canonical_id_convert_dict),part_canonical_gff_path)
            get_subfasta_main(output_region_fasta_root,region_id_path,fasta_path)
            os.system("rm {}".format(fasta_path+".fai"))
            os.system("samtools faidx {}".format(fasta_path))
            get_sub_region_table_main(region_table_path,region_id_path,id_source,part_region_table_path)
            gff_analysis_main(part_canonical_gff_path,fasta_path,canonical_stats_root,chrom_source=id_source,
                              region_table_path=part_region_table_path)
            gff_analysis_main(part_gff_path,fasta_path,rna_stats_root,chrom_source=id_source,
                              region_table_path=part_region_table_path)

        write_json(region_num,region_num_path)
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program export texts which have region ids be split")
    parser.add_argument("--processed_root",help='The root of processed data',required=True)
    parser.add_argument("--fai_path",required=True)
    parser.add_argument("--output_root",required=True)
    parser.add_argument("--fold_num",type=int,default=None,help="If it is None, then dataset would be "
                        "split by chromosome and strand, otherwise"
                        " it would be split to specific number of datasets.")
    parser.add_argument("--treat_strand_independent",action='store_true',
                        help="Each strand on training and validation dataset"
                        "would be treat independently")
    parser.add_argument("--on_double_strand_data",action='store_true')
    args = parser.parse_args()
    main(**vars(args))
