import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import read_bed, write_bed


REGION_COLUMNS = ["ordinal_id_wo_strand","ordinal_id_with_strand","coord","chr","strand","start","end"]

def get_coord(bed,use_strand=False):
    bed = bed.copy()
    query_columns = ['chr', 'start', 'end']
    if use_strand:
        query_columns.append('strand')
    coord = None
    for key in query_columns:
        if coord is None:
            coord = bed[key].astype(str)
        else:
            coord += "_" + bed[key].astype(str)
    return coord

def get_relation(bed, use_strand=True,id_prefix=None):
    """Get relation dictionary between coodinate id and oridinal id"""
    index_length = len(str(len(bed)))
    if id_prefix is None:
        SEQ_ID = "{index:0>" + str(index_length) + "d}"
    else:
        SEQ_ID = str(id_prefix) + "_{index:0>" + str(index_length) + "d}"

    query_columns = ['chr', 'start', 'end']
    if use_strand:
        query_columns.append('strand')
    bed = bed.copy().sort_values(by=query_columns)
    coords = list(get_coord(bed,use_strand=use_strand))
    index = 0
    ordinal_ids = []
    relations = {}
    used_coords = set()
    for coord in sorted(coords):
        if coord not in used_coords:
            index += 1
            used_coords.add(coord)
        ordinal_id = SEQ_ID.format(index=index)
        ordinal_ids.append(ordinal_id)
        relations[coord] = ordinal_id
    return relations


def get_region_table(bed):
    bed = bed[['chr', 'strand', 'score', 'start', 'end','id']].drop_duplicates()
    relation_with_strand = get_relation(bed,id_prefix="ss_region",use_strand=True)
    relation_wo_strand = get_relation(bed,id_prefix="ds_region",use_strand=False)
    table = bed.copy()
    table['coord'] = get_coord(bed,use_strand=True)
    table['ordinal_id_wo_strand'] = get_coord(bed,use_strand=False).replace(relation_wo_strand)
    table['ordinal_id_with_strand'] = table['coord'].replace(relation_with_strand)
    table = table[REGION_COLUMNS]
    table = table.sort_values(['ordinal_id_with_strand'])
    return table
    

def read_region_table(path, calculate_length=True):
    """Get region table about regions"""
    df = pd.read_csv(path,sep='\t',dtype={'chr': str,'start': int,'end': int})
    if calculate_length:
        df['length'] = df['end'] - df['start'] + 1
    return df


def write_region_table(table,path):
    table = table[REGION_COLUMNS]
    table.to_csv(path, index=None, sep='\t')


def main(bed_path,region_table_path,renamed_bed_path):
    bed = read_bed(bed_path)
    renamed_bed = bed.copy()
    region_table = get_region_table(bed)    
    renamed_bed['id'] = get_coord(bed,use_strand=True)
    write_bed(renamed_bed,renamed_bed_path)
    write_region_table(region_table,region_table_path)

    
if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser(
        description="This program will rename bed id field by cooridnate data")
    parser.add_argument("-i", "--bed_path", required=True,
                        help="Single-strand region bed file to be renamed")
    parser.add_argument("-r", "--renamed_bed_path", required=True,
                        help="Path to saved renamed bed file")
    parser.add_argument("-o", "--region_table_path", required=True,
                        help="Path to saved one-based renamed table")

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
