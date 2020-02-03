import sys
import os
import errno
import math
import json
import numpy as np
import pandas as pd
import datetime,pytz
from Bio import SeqIO
from pathlib import Path

class CONSTANT_LIST(list):
    def __init__(self, list_):
        super().__init__(list_)
        
    def __delitem__(self, index):
        raise Exception("{} cannot delete element".format(self.__class__.__name__))

    def insert(self, index, value):
        raise Exception("{} cannot insert element".format(self.__class__.__name__))

    def __setitem__(self, index, value):
        raise Exception("{} cannot set element".format(self.__class__.__name__))

class CONSTANT_DICT(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        raise Exception("{} cannot set element".format(self.__class__.__name__))

    def __delitem__(self, key):
        raise Exception("{} cannot delete element".format(self.__class__.__name__))
    
BED_COLUMNS = CONSTANT_LIST(['chr','start','end','id','score','strand','thick_start','thick_end',
               'rgb','count','block_size','block_related_start'])
GFF_COLUMNS = CONSTANT_LIST(['chr','source','feature','start','end','score','strand','frame','attribute'])
BASIC_GENE_MAP = CONSTANT_DICT({'gene':['exon','intron'],'other':['other']})
BASIC_GENE_ANN_TYPES = CONSTANT_LIST(['exon','intron','other'])

def get_gff_with_feature_coord(gff):
    part_gff = gff[['feature','chr','strand','start','end']]
    feature_coord = part_gff.apply(lambda x: '_'.join([str(item) for item in x]), axis=1)
    gff = gff.assign(feature_coord=feature_coord)
    return gff

def logical_not(lhs, rhs):
    return np.logical_and(lhs,np.logical_not(rhs))

def create_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as erro:
        if erro.errno != errno.EEXIST:
            raise

def get_protected_attrs_names(object_):
    class_name = object_.__class__.__name__
    attrs = [attr for attr in dir(object_) if attr.startswith('_')
             and not attr.endswith('__')
             and not attr.startswith('_'+class_name+'__')]
    return attrs

def split(ids,ratios):
    if round(sum(ratios))!=1:
        raise Exception("Ratio sum should be one")
    lb = ub = 0
    ids_list=[]
    id_len = len(ids)
    sum_=0
    for ratio in ratios:
        ub += ratio
        item = ids[math.ceil(lb*id_len):math.ceil(ub*id_len)]
        sum_+=len(item)
        ids_list.append(item)
        lb=ub
    if sum_!=id_len:
        raise Exception("Id number is not consist with origin count")
    return ids_list

def get_subdict(ids,data):
    return dict(zip(ids,[data[id_] for id_ in ids]))

def read_bed(path,convert_to_one_base=True):
    #Read bed data into one-based DataFrame format
    try:
        bed = pd.read_csv(path,sep='\t',header=None,dtype={0:str})
    except:
        raise Exception("{} has incorrect format".format(path))
    bed.columns = BED_COLUMNS[:len(bed.columns)]
    data = []
    for item in bed.to_dict('record'):
        temp = dict(item)
        if temp['strand'] == '+':
            temp['five_end'] = temp['start']
            if convert_to_one_base:
                temp['five_end'] += 1
            temp['three_end'] = temp['end']
        else:
            temp['five_end'] = temp['end']
            temp['three_end'] = temp['start']
            if convert_to_one_base:
                temp['three_end'] += 1
                
        data.append(temp)
    df = pd.DataFrame.from_dict(data)
    df = df.astype(str)
    for name in ['start','end','thick_start','thick_end','count']:
        if name in df.columns:
            df[name] = df[name].astype(float).astype(int)
            if name in ['start','thick_start'] and convert_to_one_base:
                df[name] += 1
    return df

def write_bed(bed,path,from_one_base=True):
    columns = []
    for name in BED_COLUMNS:
        if name in bed.columns:
            columns.append(name) 
    bed = bed[columns].astype(str)
    for name in ['start','end','thick_start','thick_end','count']:
        if name in bed.columns:
            bed[name] = bed[name].astype(float).astype(int)
            if name in ['start','thick_start'] and from_one_base:
                bed[name] -= 1
    bed.to_csv(path,sep='\t',index=None,header=None)

def read_gff(path):
    gff = pd.read_csv(path,sep='\t',header=None,names=list(range(len(GFF_COLUMNS))),dtype=str)
    gff.columns = GFF_COLUMNS
    gff = gff[~gff['chr'].str.startswith('#')]
    int_columns = ['start','end']
    gff.loc[:,int_columns] = gff[int_columns].astype(float).astype(int)
    return gff
    
def write_gff(gff,path):
    fp = open(path, 'w')
    fp.write("##gff-version 3\n")
    gff[GFF_COLUMNS].to_csv(fp,header=None,sep='\t',index=None)
    fp.close()

def get_gff_item_with_attribute(item,split_attr=None):
    split_attr = split_attr or []
    attributes = item['attribute'].split(';')
    attribute_dict = {}
    for attribute in attributes:
        lhs,rhs = attribute.split('=')
        lhs = lhs.lower()
        if lhs in split_attr:
            rhs = rhs.split(',')
        attribute_dict[lhs] = rhs
    copied = dict(item)
    copied.update(attribute_dict)
    return copied
    
def get_gff_with_attribute(gff,split_attr=None):
    df_dict = gff.to_dict('record')
    data = []
    for item in df_dict:
        data += [get_gff_item_with_attribute(item,split_attr)]
    gff = pd.DataFrame.from_dict(data)
    return gff
    
def dupliacte_gff_by_parent(gff):
    preprocessed = []
    for item in gff.to_dict('record'):
        parents = item['parent']
        #If parent is not NaN
        if parents == parents:
            for parent in parents:
                item_ = dict(item)
                item_['parent'] = str(parent)
                preprocessed.append(item_)
        else:
            preprocessed.append(item)
    gff = pd.DataFrame.from_dict(preprocessed)
    return gff
    
def read_fai(path):
    chrom_info = pd.read_csv(path,header=None,sep='\t')
    chrom_id, chrom_length = chrom_info[0], chrom_info[1]
    chrom_info = {}
    for id_,length in zip(chrom_id,chrom_length):
        chrom_info[str(id_)] = length
    return chrom_info

def read_gffcompare_stats(stats_path):
    """Return sensitivity, precision and miss_novel_stats"""
    sensitivity = {}
    precision = {}
    miss_novel_stats = {}
    with open(stats_path,'r') as fp:
        for line in fp:
            if not line.startswith("#"):
                if 'level' in line:
                    title,values = line.split(':')
                    title = title.lower().strip()
                    values = values.split('|')[:2]
                    sensitivity[title] = float(values[0])
                    precision[title] = float(values[1])
                if 'Missed' in line or 'Novel' in line:
                    title,values = line.split(':')
                    title = title.lower().strip()
                    values = float(values.split('(')[1][:-3])
                    miss_novel_stats[title] = values
        return sensitivity,precision,miss_novel_stats
    
def read_fasta(paths):
    """Read fasta file and return dictionary of sequneces"""
    if not isinstance(paths,list):
        paths = [paths]
    data = {}
    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        with open(path) as file:
            fasta_sequences = SeqIO.parse(file, 'fasta')
            for fasta in fasta_sequences:
                name, seq = fasta.id, str(fasta.seq)
                data[name]=seq
    return data

def write_fasta(path,seqs):
    """Read dictionary of sequneces into fasta file"""
    with open(path,"w") as file:
        for id_,seq in seqs.items():
            file.write(">" + id_ + "\n")
            file.write(seq + "\n")

def gffcompare_command(answer_gff_path,predict_gff_path,prefix_path,merge=False,verbose=False):
    gffcompare_command = '$GFFCOMPARE --strict-match --chr-stats --debug -T -e 0 -d 0 -r {} {} -o {}'
    if not merge:
        gffcompare_command += ' --no-merge'
    command = gffcompare_command.format(answer_gff_path,predict_gff_path,prefix_path)
    if verbose:
        print(command)
    os.system(command)
    
def print_progress(info):
    print(info,end='\r')
    sys.stdout.write('\033[K')
    
def write_json(json_,path,mode=None):
    mode = mode or 'w'
    with open(path,mode) as fp:
        json.dump(json_, fp, indent=4,sort_keys=True)

def read_json(path,mode=None):
    mode = mode or 'r'
    with open(path,mode) as fp:
        return json.load(fp)

def read_region_table(path,calculate_length=True):
    """Get region table about regions"""
    columns = ['new_id','old_id','chr','strand','start','end']
    df = pd.read_csv(path,sep='\t',dtype={'chr':str,'start':int,'end':int})
    if calculate_length:
        df['length'] = df['end'] - df['start'] + 1
        columns += ['length']
    return df[columns]

def get_time_str(time_zone=None):
    time_zone = time_zone or pytz.timezone('ROC')
    time_str = datetime.datetime.now(time_zone).strftime("%Y-%m-%d %H:%M:%S %z")
    return time_str
