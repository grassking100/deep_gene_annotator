import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_gff,get_gff_with_attribute


def main(input_gff_path,non_hypothetical_gene_id_path,valid_chroms=None):
    gff = read_gff(input_gff_path)
    if valid_chroms is not None:
        gff = gff[gff['chr'].isin(valid_chroms)]
    gff = get_gff_with_attribute(gff)
    coding_gene = gff[gff['locus_type'] == 'protein_coding']
    non_hypothetical_gene = coding_gene[(~coding_gene['note'].str.startswith('hypothetical'))]
    non_hypothetical_gene_ids = set(non_hypothetical_gene['id'])
    with open(non_hypothetical_gene_id_path,'w') as fp:
        for id_ in non_hypothetical_gene_ids:
            fp.write("{}\n".format(id_))

if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_gff_path", required=True)
    parser.add_argument("-o", "--non_hypothetical_gene_id_path", required=True)
    parser.add_argument("-v","--valid_chroms",
                        type=lambda x: x.split(','))
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
