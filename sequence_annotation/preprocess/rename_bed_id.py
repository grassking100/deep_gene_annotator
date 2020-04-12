import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_bed, write_bed

def get_bed_with_coord(bed, use_strand=False,
                       viewer_format=False,
                       id_prefix=None):
    bed = bed.copy()
    index_length = len(str(len(bed)))
    if id_prefix is None:
        SEQ_ID = "{index:0>" + str(index_length) + "d}"
    else:
        SEQ_ID = str(id_prefix) + "_{index:0>" + str(index_length) + "d}"

    query_columns = ['chr', 'start', 'end']
    if use_strand:
        query_columns.append('strand')
    if viewer_format:
        coord_id = bed['chr'].astype(str) + ":" + bed['start'].astype(
            str) + "-" + bed['end'].astype(str)
        if use_strand:
            coord_id = coord_id + "(" + bed['strand'] + ")"
    else:
        coord_id = None
        for key in query_columns:
            if coord_id is None:
                coord_id = bed[key].astype(str)
            else:
                coord_id += "_" + bed[key].astype(str)
    bed = bed.assign(coord_id=coord_id)
    bed = bed.sort_values(by=query_columns)
    index = 0
    record_coord_ids = set()
    ordinal_ids = []
    for coord_id in list(bed['coord_id']):
        if coord_id not in record_coord_ids:
            index += 1
            record_coord_ids.add(coord_id)
        ordinal_id = SEQ_ID.format(index=index)
        ordinal_ids.append(ordinal_id)
    bed = bed.assign(ordinal_id=ordinal_ids)
    return bed


def get_region_table(bed_with_coord_id, new_id_source, record_coord_id=False):
    bed_with_coord_id = bed_with_coord_id.copy()
    bed_with_coord_id['new_id'] = bed_with_coord_id[new_id_source]
    if record_coord_id:
        bed_with_coord_id['old_id'] = bed_with_coord_id['coord_id']
    else:
        bed_with_coord_id['old_id'] = bed_with_coord_id['id']
    bed_with_coord_id = pd.DataFrame.from_dict(bed_with_coord_id).sort_values(
        by=['new_id'])
    bed_with_coord_id = bed_with_coord_id[[
        'new_id', 'old_id', 'chr', 'strand', 'start', 'end'
    ]]
    return bed_with_coord_id


def get_renamed_bed(bed, new_id_source):
    bed = bed.copy()
    bed['id'] = bed[new_id_source]
    bed = bed.sort_values(by=['id'])
    return bed


def main(bed_path, saved_table_path, renamed_bed_path, 
         new_id_source, record_coord_id=False,
         ignore_output_strand=False, id_prefix=None,
         viewer_format=False, use_strand=False):
    try:
        bed = read_bed(bed_path)
    except BaseException:
        raise Exception("Wrong format in {}".format(bed_path))

    print("Get bed with cooridnate id")
    bed_with_coord_id = get_bed_with_coord(bed,
                                           id_prefix=id_prefix,
                                           use_strand=use_strand,
                                           viewer_format=viewer_format)
    print("Rename bed")
    renamed_bed = get_renamed_bed(bed_with_coord_id,
                                  new_id_source=new_id_source)
    if ignore_output_strand:
        renamed_bed['strand'] = '.'
    renamed_bed = renamed_bed[['chr', 'strand', 'score', 'start', 'end',
                               'id']].drop_duplicates()
    write_bed(renamed_bed, renamed_bed_path)
    print("Get renamed table")
    region_table = get_region_table(bed_with_coord_id,
                                    new_id_source,
                                    record_coord_id=record_coord_id)
    region_table.to_csv(saved_table_path, index=None, sep='\t')


if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser(
        description="This program will rename bed id field by cooridnate data")
    parser.add_argument("-i", "--bed_path", required=True,
                        help="Bed file to be renamed")
    parser.add_argument("-o", "--renamed_bed_path", required=True,
                        help="Path to saved renamed bed file")
    parser.add_argument("-t", "--saved_table_path", required=True,
                        help="Path to saved one-based renamed table")
    parser.add_argument("-s", "--use_strand", action='store_true')
    parser.add_argument("-p", "--id_prefix", help="Prefix of new id")
    parser.add_argument("--new_id_source",
                        choices=['coord_id', 'ordinal_id'],
                        default='ordinal_id')
    parser.add_argument("--record_coord_id",action='store_true',
                        help='Use coordinate as old id')
    parser.add_argument("--ignore_output_strand", action='store_true')
    parser.add_argument("--viewer_format",action='store_true',
                        help='If it is true then id would be like '
                        'chr:start-end(strand), otherwise it would '
                        'be lile chr_strand_start_end')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
