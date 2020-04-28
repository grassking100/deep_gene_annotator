import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.preprocess.utils import get_gff_with_belonging
from sequence_annotation.utils.utils import get_gff_with_attribute, get_gff_with_updated_attribute
from sequence_annotation.utils.utils import read_gff, write_gff, dupliacte_gff_by_parent


def main(input_gff_path,output_gff_path,valid_chroms):
    gff = read_gff(input_gff_path)
    gff.loc[:, 'chr'] = gff['chr'].str.replace('Chr', '')
    gff = gff[gff['chr'].isin(valid_chroms)]
    gff = get_gff_with_attribute(gff, ['parent'])
    gff = dupliacte_gff_by_parent(gff)
    gff = get_gff_with_belonging(gff)
    discarded_ids = list(gff[gff['feature'].isin(['pre_miRNA','miRNA'])]['id'])
    gff = gff[~gff['id'].isin(discarded_ids)]
    gff = get_gff_with_updated_attribute(gff)
    write_gff(gff, output_gff_path)


if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser(description='Remove miRNA, and pre_miRNA '
                            'related data, rename chromomsome id and '
                            'dupliacte gff item by their parent id')
    parser.add_argument("-i", "--input_gff_path", required=True)
    parser.add_argument("-o", "--output_gff_path", required=True)
    parser.add_argument("-v","--valid_chroms",required=True,
                        type=lambda x: x.split(','))
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
