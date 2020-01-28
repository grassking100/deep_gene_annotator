import os,sys
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_gff,write_gff
from sequence_annotation.utils.utils import get_gff_with_attribute

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_gff_path",required=True)
    parser.add_argument("-o", "--output_gff_path",required=True)
    
    args = parser.parse_args()
    gff = read_gff(args.input_gff_path)
    gff = get_gff_with_attribute(gff)
    alt_site_gene_id = set(gff[gff['feature'].isin(['alt_donor','alt_acceptor'])]['parent'])
    cleaned_gff = gff[(~gff['parent'].isin(alt_site_gene_id))&(~gff['id'].isin(alt_site_gene_id))]
    write_gff(cleaned_gff,args.output_gff_path)
