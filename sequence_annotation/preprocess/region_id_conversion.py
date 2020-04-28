import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_bed, write_bed

def get_coord(bed,use_strand=True):
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

def get_relation(bed_with_coord, use_strand=True,
                 id_prefix=None):
    """Get relation between coodinate id and oridinal id"""
    bed = bed_with_coord.copy()
    index_length = len(str(len(bed)))
    if id_prefix is None:
        SEQ_ID = "{index:0>" + str(index_length) + "d}"
    else:
        SEQ_ID = str(id_prefix) + "_{index:0>" + str(index_length) + "d}"

    query_columns = ['chr', 'start', 'end']
    if use_strand:
        query_columns.append('strand')
    bed = bed.sort_values(by=query_columns)
    if use_strand:
        coords = list(bed['strand_coord'])
    else:
        coords = list(bed['coord'])
    index = 0
    ordinal_ids = []
    relations = {}
    used_coords = set()
    for coord in coords:
        if coord not in used_coords:
            index += 1
            used_coords.add(coord)
        ordinal_id = SEQ_ID.format(index=index)
        ordinal_ids.append(ordinal_id)
        relations[coord] = ordinal_id
    return relations


def get_region_table(bed_with_coord, relation_with_strand, relation_wo_strand):
    table = bed_with_coord.copy()
    table['ordinal_id_with_strand'] = table['strand_coord']
    table['ordinal_id_wo_strand'] = table['coord']
    table['ordinal_id_with_strand'] = table['ordinal_id_with_strand'].replace(relation_with_strand)
    table['ordinal_id_wo_strand'] = table['ordinal_id_wo_strand'].replace(relation_wo_strand)
    table = table[['ordinal_id_wo_strand','ordinal_id_with_strand',
                   'strand_coord','chr', 'strand', 'start', 'end']]
    table = table.rename(columns={'strand_coord':'coord'})
    return table


def main(bed_path,coord_region_bed_path,
         region_table_path,id_prefix=None):
    try:
        bed = read_bed(bed_path)
    except BaseException:
        raise Exception("Wrong format in {}".format(bed_path))

    print("Get bed with coordinate id")
    bed = bed[['chr', 'strand', 'score', 'start', 'end','id']].drop_duplicates()
    bed_with_coord = bed.copy()
    bed_with_coord['strand_coord'] = get_coord(bed,use_strand=True)
    bed_with_coord['coord'] = get_coord(bed_with_coord,use_strand=False)
    relation_with_strand = get_relation(bed_with_coord,id_prefix=id_prefix,
                                        use_strand=True)
    relation_wo_strand = get_relation(bed_with_coord,id_prefix=id_prefix,
                                      use_strand=False)
    table = get_region_table(bed_with_coord, relation_with_strand, relation_wo_strand)
    print("Rename bed")
    coord_bed = bed_with_coord.copy()
    coord_bed['id'] = coord_bed['strand_coord']
    write_bed(coord_bed.drop_duplicates(),coord_region_bed_path)
    table.to_csv(region_table_path, index=None, sep='\t')

    

if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser(
        description="This program will rename bed id field by cooridnate data")
    parser.add_argument("-i", "--bed_path", required=True,
                        help="Single-strand region bed file to be renamed")
    parser.add_argument("-c", "--coord_region_bed_path", required=True,
                        help="Path to saved coprdinate single-strand bed file")
    parser.add_argument("-t", "--region_table_path", required=True,
                        help="Path to saved one-based renamed table")
    parser.add_argument("-p", "--id_prefix", help="Prefix of new id")

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
