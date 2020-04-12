import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_gff, write_gff, read_bed, write_bed
from sequence_annotation.preprocess.utils import read_region_table


def rename_chrom(data, renamed_table):
    belong_table = {}
    for item in renamed_table.to_dict('record'):
        id_ = item['old_id']
        belong_table[id_] = item['new_id']
    data = data.copy()
    data['chr'] = data['chr'].replace(belong_table)
    return data


def main(input_path, table_path, output_path):
    renamed_table = read_region_table(table_path)
    if 'gff' in input_path.split('.')[-1]:
        data = read_gff(input_path)
    else:
        data = read_bed(input_path)
    redefined = rename_chrom(data, renamed_table)
    if 'gff' in input_path.split('.')[-1]:
        write_gff(redefined, output_path)
    else:
        write_bed(redefined, output_path)


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename chromosome")
    parser.add_argument("-i",
                        "--input_path",
                        help="Path of gff or bed file to renamed chromosome",
                        required=True)
    parser.add_argument("-t",
                        "--table_path",
                        help="Table about renamed region",
                        required=True)
    parser.add_argument("-o",
                        "--output_path",
                        help="Path to saved renamed file",
                        required=True)

    args = parser.parse_args()
    main(args.input_path, args.table_path, args.output_path)
