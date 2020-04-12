import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_gff, get_gff_with_attribute
from sequence_annotation.preprocess.utils import RNA_TYPES


def get_id_table(gff):
    transcripts = gff[gff['feature'].isin(RNA_TYPES)]
    transcripts = get_gff_with_attribute(transcripts).to_dict('record')

    ids = []
    for transcript in transcripts:
        item = {}
        item['gene_id'] = transcript['parent']
        item['transcript_id'] = transcript['id']
        ids.append(item)

    id_table = pd.DataFrame.from_dict(ids).drop_duplicates()
    return id_table


def read_id_table(path):
    return pd.read_csv(path, sep='\t')


def write_id_table(table, table_path):
    table.to_csv(table_path, index=None, sep='\t')


def convert_id_table_to_dict(table):
    returned = {}
    for g_id, t_id in zip(table['gene_id'], table['transcript_id']):
        returned[t_id] = g_id
    return returned


def get_id_convert_dict(id_table_path):
    id_table = read_id_table(id_table_path)
    return convert_id_table_to_dict(id_table)


def main(gff_path, table_path):
    gff = read_gff(gff_path)
    id_table = get_id_table(gff)
    write_id_table(id_table, table_path)


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--gff_path", required=True)
    parser.add_argument("-o", "--table_path", required=True)
    args = parser.parse_args()

    main(**vars(args))
