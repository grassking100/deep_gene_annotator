import pytz
import datetime
import re
import sys
import os
import errno
import math
import json
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from .exception import InvalidStrandType
pd.set_option('mode.chained_assignment', 'raise')


class CONSTANT_LIST(list):
    def __init__(self, list_):
        super().__init__(list_)

    def __delitem__(self, index):
        raise Exception("{} cannot delete element".format(
            self.__class__.__name__))

    def insert(self, index, value):
        raise Exception("{} cannot insert element".format(
            self.__class__.__name__))

    def __setitem__(self, index, value):
        raise Exception("{} cannot set element".format(
            self.__class__.__name__))


class CONSTANT_DICT(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        raise Exception("{} cannot set element".format(
            self.__class__.__name__))

    def __delitem__(self, key):
        raise Exception("{} cannot delete element".format(
            self.__class__.__name__))


BED_COLUMNS = CONSTANT_LIST([
    'chr', 'start', 'end', 'id', 'score', 'strand', 'thick_start', 'thick_end',
    'rgb', 'count', 'block_size', 'block_related_start'
])
GFF_COLUMNS = CONSTANT_LIST([
    'chr', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame',
    'attribute'
])
BASIC_GENE_MAP = CONSTANT_DICT({
    'gene': ['exon', 'intron'],
    'other': ['other']
})
BASIC_GENE_ANN_TYPES = CONSTANT_LIST(['exon', 'intron', 'other'])

BASIC_SIMPLIFY_MAP = CONSTANT_DICT({
    'exon': ['exon'],
    'intron': ['intron'],
    'other': ['other']
})

BASIC_COLOR_SETTING = CONSTANT_DICT({
    'other': 'blue',
    'exon': 'red',
    'intron': 'yellow'
})

def get_gff_with_feature_coord(gff):
    gff = gff.copy()
    part_gff = gff[['feature', 'chr', 'strand', 'start', 'end']]
    feature_coord = part_gff.apply(
        lambda x: '_'.join([str(item) for item in x]), axis=1)
    gff = gff.assign(feature_coord=feature_coord)
    return gff


def logical_not(lhs, rhs):
    return np.logical_and(lhs, np.logical_not(rhs))


def create_folder(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError as erro:
        if erro.errno != errno.EEXIST:
            raise


def copy_path(root, path):
    target_path = os.path.join(root, path.split('/')[-1])
    if not os.path.exists(target_path):
        command = 'cp -t {} {}'.format(root, path)
        os.system(command)


def get_protected_attrs_names(object_):
    class_name = object_.__class__.__name__
    attrs = [
        attr for attr in dir(object_)
        if attr.startswith('_') and not attr.endswith('__')
        and not attr.startswith('_' + class_name + '__')
    ]
    return attrs


def split(ids, ratios):
    if round(sum(ratios)) != 1:
        raise Exception("Ratio sum should be one")
    lb = ub = 0
    ids_list = []
    id_len = len(ids)
    sum_ = 0
    for ratio in ratios:
        ub += ratio
        item = ids[math.ceil(lb * id_len):math.ceil(ub * id_len)]
        sum_ += len(item)
        ids_list.append(item)
        lb = ub
    if sum_ != id_len:
        raise Exception("Id number is not consist with origin count")
    return ids_list


def get_subdict(ids, data):
    return dict(zip(ids, [data[id_] for id_ in ids]))


def validate_bed(bed):
    if len(bed) > 0:
        if len(set(bed['strand']) - set(['+', '-', '.'])) > 0:
            raise InvalidStrandType()
        if ((bed['end'] - bed['start'] + 1) <= 0).any():
            raise Exception("Wrong transcript size")

        if 'start' in bed and (bed['start']<=0).any():
            raise Exception("Wrong transcript start")

        if 'end' in bed and (bed['end']<=0).any():
            raise Exception("Wrong transcript end")
            
        if 'thick_end' in bed.columns and 'thick_start' in bed.columns:
            if ((bed['thick_end'] - bed['thick_start'] + 1) < 0).any():
                raise Exception("Wrong coding size")
        if 'block_size' in bed.columns:
            block_size_list = list(bed['block_size'])
            for block_sizes in block_size_list:
                block_size_status = [
                    int(size) <= 0 for size in block_sizes.split(',')
                ]
                if any(block_size_status):
                    raise Exception("Wrong block size")

        if 'block_related_start' in bed.columns:
            block_related_start_list = list(bed['block_related_start'])
            for block_related_starts in block_related_start_list:
                site_status = [
                    int(site) < 0 for site in block_related_starts.split(',')
                ]
                if any(site_status):
                    raise Exception("Wrong block start size")


def validate_gff(gff):
    if len(gff) > 0:
        if len(set(gff['strand']) - set(['+', '-', '.'])) > 0:
            raise InvalidStrandType()
        if ((gff['end'] - gff['start'] + 1) <= 0).any():
            raise Exception("Wrong block size")
        if any(gff['start']<=0):
            raise Exception("Wrong block start")
        if any(gff['end']<=0):
            raise Exception("Wrong block end")

def read_bed(path):
    """
    Read bed data and convert from interbase coordinate system（ICS）to base coordinate system(BCS)
    For more information, please visit https://tidyomics.com/blog/2018/12/09/2018-12-09-the-devil-0-and-1-coordinate-system-in-genomics
    """
    try:
        bed = pd.read_csv(path, sep='\t', header=None, dtype={0: str,10:str,11:str})
    except BaseException:
        raise Exception("{} has incorrect format".format(path))
    bed.columns = BED_COLUMNS[:len(bed.columns)]
    data = []
    for item in bed.to_dict('record'):
        temp = dict(item)
        if temp['strand'] == '+':
            temp['five_end'] = temp['start']
            temp['five_end'] += 1
            temp['three_end'] = temp['end']
        else:
            temp['five_end'] = temp['end']
            temp['three_end'] = temp['start']
            temp['three_end'] += 1

        data.append(temp)
    bed = pd.DataFrame.from_dict(data)
    bed = bed.astype(str)
    for name in ['start', 'end', 'thick_start', 'thick_end', 'count']:
        if name in bed.columns:
            bed[name] = bed[name].astype(float).astype(int)
            if name in ['start', 'thick_start']:
                bed[name] += 1
    validate_bed(bed)
    return bed


def write_bed(bed, path):
    """
    Convert bed data from base coordinate system(BCS) to interbase coordinate system（ICS) and write to file
    For more information, please visit https://tidyomics.com/blog/2018/12/09/2018-12-09-the-devil-0-and-1-coordinate-system-in-genomics
    """
    validate_bed(bed)
    columns = []
    for name in BED_COLUMNS:
        if name in bed.columns:
            columns.append(name)
    bed = bed[columns].astype(str)
    for name in ['start', 'end', 'thick_start', 'thick_end', 'count']:
        if name in bed.columns:
            bed[name] = bed[name].astype(float).astype(int)
            if name in ['start', 'thick_start']:
                bed[name] -= 1
    if len(bed) > 0 and len(set(bed['strand']) - set(['+', '-', '.'])) > 0:
        raise InvalidStrandType()
    bed.to_csv(path, sep='\t', index=None, header=None)


def read_gff(path):
    gff = pd.read_csv(path,
                      sep='\t',
                      header=None,
                      names=list(range(len(GFF_COLUMNS))),
                      dtype=str)
    gff.columns = GFF_COLUMNS
    gff = gff[~gff['chr'].str.startswith('#')]
    int_columns = ['start', 'end']
    gff.loc[:, int_columns] = gff[int_columns].astype(float).astype(int)
    validate_gff(gff)
    return gff


def write_gff(gff, path):
    validate_gff(gff)
    fp = open(path, 'w')
    fp.write("##gff-version 3\n")
    gff[GFF_COLUMNS].to_csv(fp, header=None, sep='\t', index=None)
    fp.close()


def get_gff_item_with_attribute(item, split_attr=None):
    split_attr = split_attr or []
    attributes = item['attribute'].split(';')
    attribute_dict = {}
    for attribute in attributes:
        lhs, rhs = attribute.split('=')
        lhs = lhs.lower().replace('-', '_')
        if lhs in split_attr:
            rhs = rhs.split(',')
        attribute_dict[lhs] = rhs
    copied = dict(item)
    copied.update(attribute_dict)
    return copied


def get_gff_with_attribute(gff, split_attr=None):
    df_dict = gff.to_dict('record')
    data = []
    for item in df_dict:
        data += [get_gff_item_with_attribute(item, split_attr)]
    gff = pd.DataFrame.from_dict(data)
    gff = gff.where(pd.notnull(gff), None)
    return gff


def get_gff_with_updated_attribute(gff):
    gff = gff.copy()
    columns = [c for c in gff.columns if c not in GFF_COLUMNS]
    attributes = []
    for column in columns:
        if column.lower() == 'id':
            key = 'ID'
        else:
            key = column.capitalize()
        values = gff[column]
        attribute = values.apply(lambda value: "{}={}".format(key, value))
        attributes += [attribute]
    if len(attributes) > 0:
        attribute = attributes[0]
        for attr in attributes[1:]:
            attribute = attribute + ";" + attr
        gff['attribute'] = attribute.replace(
            r"(;\w*=(None|\.))|(^\w*=(None|\.);)|(^\w*=(None|\.)$)",
            '',
            regex=True)
    return gff


def dupliacte_gff_by_parent(gff):
    if 'parent' not in gff.columns:
        raise Exception("GFF file lacks 'parent' column")

    valid_parents = [p for p in gff['parent'] if p is not None]
    if len(valid_parents) > 0:
        if not isinstance(valid_parents[0], list):
            raise Exception("GFF's 'parent' data type should be list")

    preprocessed = []
    for item in gff.to_dict('record'):
        parents = item['parent']
        if parents is not None:
            for parent in parents:
                item_ = dict(item)
                item_['parent'] = str(parent)
                preprocessed.append(item_)
        else:
            preprocessed.append(item)
    gff = pd.DataFrame.from_dict(preprocessed)
    return gff


def read_fai(path):
    chrom_info = pd.read_csv(path, header=None, sep='\t')
    chrom_id, chrom_length = chrom_info[0], chrom_info[1]
    chrom_info = {}
    for id_, length in zip(chrom_id, chrom_length):
        chrom_info[str(id_)] = length
    return chrom_info


def read_gffcompare_stats(stats_path):
    """Return sensitivity, precision and miss_novel_stats"""
    sensitivity = {}
    precision = {}
    miss_novel_stats = {}
    with open(stats_path, 'r') as fp:
        for line in fp:
            if not line.startswith("#"):
                if 'level' in line:
                    title, values = line.split(':')
                    title = title.lower().strip()
                    values = values.split('|')[:2]
                    sensitivity[title] = float(values[0])
                    precision[title] = float(values[1])
                if 'Missed' in line or 'Novel' in line:
                    title, values = line.split(':')
                    title = title.lower().strip()
                    values = float(values.split('(')[1][:-3])
                    miss_novel_stats[title] = values
        return sensitivity, precision, miss_novel_stats


def read_fasta(path, check_unique_id=True):
    """Read fasta file and return dictionary of sequneces"""
    data = {}
    if not Path(path).exists():
        raise FileNotFoundError(path)
    with open(path) as file:
        fasta_sequences = SeqIO.parse(file, 'fasta')
        for fasta in fasta_sequences:
            name, seq = fasta.id, str(fasta.seq)
            if check_unique_id and name in data:
                raise Exception("Duplicate id {}".format(name))
            data[name] = seq
    return data


def write_fasta(seqs,path):
    """Read dictionary of sequneces into fasta file"""
    with open(path, "w") as file:
        for id_, seq in seqs.items():
            file.write(">" + id_ + "\n")
            file.write(seq + "\n")


def gffcompare_command(answer_gff_path,
                       predict_gff_path,
                       prefix_path,
                       merge=False,
                       verbose=False):
    gffcompare_command = '$GFFCOMPARE --strict-match --chr-stats --debug -T -e 0 -d 0 -r {} {} -o {}'
    if not merge:
        gffcompare_command += ' --no-merge'
    command = gffcompare_command.format(answer_gff_path, predict_gff_path,
                                        prefix_path)
    if verbose:
        print(command)
    os.system(command)


def print_progress(info):
    print(info, end='\r')
    sys.stdout.write('\033[K')


def write_json(json_, path, mode=None):
    mode = mode or 'w'
    with open(path, mode) as fp:
        json.dump(json_, fp, indent=4, sort_keys=True)


def read_json(path, mode=None):
    mode = mode or 'r'
    with open(path, mode) as fp:
        return json.load(fp)


def replace_utc_to_local(timestamp, timezone_=None):
    timezone_ = timezone_ or pytz.timezone('ROC')
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=pytz.timezone('UTC'))
    return timestamp.astimezone(timezone_)


def get_time(timezone_=None):
    timezone_ = timezone_ or pytz.timezone('ROC')
    time_data = datetime.datetime.now(timezone_)
    return time_data


def to_time_str(time_data):
    time_str = time_data.strftime("%Y-%m-%d %H:%M:%S.%f %z")
    return time_str


def from_time_str(time_str):
    time_data = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f %z")
    return time_data


def get_time_str(timezone_=None):
    time_str = to_time_str(get_time(timezone_))
    return time_str


def get_file_name(path, with_postfix=False):
    name = path.split('/')[-1]
    if not with_postfix:
        name = '.'.join(name.split('.')[:-1])
    return name


def batch_join(project_root, folder_names, path):
    paths = {}
    for folder_name in folder_names:
        paths[folder_name] = os.path.join(project_root, folder_name, path)
    return paths


def find_substr(regex, string, shift_value=None):
    """Find indice of matched text in string

    Parameters:
    ----------
    regex : str
        Regular expression
    string : str
        String to be searched
    shift_value : int (default: 0)
        Shift the index

    Returns:
    ----------
    list (int)
        List of indice
    """
    shift_value = shift_value or 0
    iter_ = re.finditer(regex, string)
    indice = [m.start() + shift_value for m in iter_]
    return indice
