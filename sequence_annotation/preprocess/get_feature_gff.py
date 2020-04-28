import sys
import os
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_gff, write_gff
from sequence_annotation.preprocess.utils import GENE_TYPES, RNA_TYPES, EXON_TYPES, SUBEXON_TYPES

TYPES = {'GENE':GENE_TYPES,'RNA':RNA_TYPES,
         'EXON':EXON_TYPES,'SUBEXON':SUBEXON_TYPES}

def main(input_path, output_path, features):
    gff = read_gff(input_path)
    if features in TYPES:
        features = TYPES[features]
    else:
        features = features.split(',')
    subgff = gff[gff['feature'].isin(features)]
    write_gff(subgff, output_path)

if __name__ == '__main__':
    parser = ArgumentParser(
        description="This program will convert gff file to bed file. "
        "It will treat alt_acceptor and alt_donor regions as intron")
    parser.add_argument("-i","--input_path",required=True,
                        help="Path of input gff file")
    parser.add_argument("-o","--output_path",required=True,
                        help="Path of output gff file")
    parser.add_argument("-f","--features",required=True,
                        help="Features with comma separated")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
