import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
from argparse import ArgumentParser
from sequence_annotation.file_process.utils import read_gff, write_gff, read_bed, write_bed
from sequence_annotation.file_process.utils import read_fasta,write_fasta
from sequence_annotation.file_process.get_region_table import read_region_table


def rename_chrom(data, region_table, source,target,type_):
    renamed = dict(zip(list(region_table[source]),list(region_table[target])))
    if type_ in ['gff3','bed']:
        new_data = data.copy()
        new_data['chr'] = new_data['chr'].replace(renamed)
    else:
        new_data = {}
        for id_,seq in data.items():
            new_data[renamed[id_]] = seq
    return new_data


def main(input_path, region_table_path, output_path,source,target):
    region_table = read_region_table(region_table_path)
    type_ = input_path.split('.')[-1]
    if 'gff3' == type_:
        data = read_gff(input_path)
    elif 'bed' == type_:
        data = read_bed(input_path)
    else:
        data = read_fasta(input_path)
    renamed = rename_chrom(data, region_table,source,target,type_)
    if 'gff3' == type_:
        write_gff(renamed, output_path)
    elif 'bed' == type_:
        write_bed(renamed, output_path)
    else:
        write_fasta(renamed, output_path)


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will rename chromosome")
    parser.add_argument("-i","--input_path",required=True,
                        help="Path of gff or bed file to renamed chromosome")
    parser.add_argument("-t","--region_table_path",required=True,
                        help="Table about renamed region")
    parser.add_argument("-o","--output_path",required=True,
                        help="Path to saved renamed file")
    parser.add_argument("--source",required=True)
    parser.add_argument("--target",required=True)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
