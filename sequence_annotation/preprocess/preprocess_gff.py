import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_gff, write_gff, get_gff_with_attribute, dupliacte_gff_by_parent

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description='Remove miRNA, uORF, protein and miRNA_primary_transcript related data and rename chromomsome id')
    parser.add_argument("-i","--input_gff_path",required=True)
    parser.add_argument("-o","--output_gff_path",required=True)
    args = parser.parse_args()

    gff = read_gff(args.input_gff_path)
    gff = dupliacte_gff_by_parent(get_gff_with_attribute(gff,['parent']))
    discared_data = gff[gff['feature'].isin(['miRNA','uORF','protein','miRNA_primary_transcript'])]
    discard_ids = list(discared_data['id'])
    discard_ids += list(discared_data['parent'])
    discard_ids = set(discard_ids)
    if None in discard_ids:
        discard_ids.remove(None)
    is_id_discard = gff['id'].isin(discard_ids)
    is_parent_discard = gff['parent'].isin(discard_ids)
    gff = gff[(~is_id_discard) & (~is_parent_discard)]
    #print(gff.head())
    gff.loc[:,'chr'] = gff['chr'].str.replace('Chr','')
    write_gff(gff,args.output_gff_path)
    