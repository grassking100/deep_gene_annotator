import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import get_gff_with_belonging
from sequence_annotation.file_process.utils import get_gff_with_attribute, get_gff_with_updated_attribute
from sequence_annotation.file_process.utils import read_gff, write_gff, dupliacte_gff_by_parent
from sequence_annotation.file_process.utils import CDS_TYPE, EXON_TYPE, GENE_TYPE, TRANSCRIPT_TYPE, UTR_TYPE
from species.arabidopsis.utils import GENE_TYPES,TRANSCRIPT_TYPES,EXON_TYPES,CDS_TYPES,FIVE_PRIME_UTR_TYPES,THREE_PRIME_UTR_TYPES
from species.arabidopsis.utils import PRIMARY_MIRNA_TPYES,MIRNA_TPYES,UORF_TYPES,PROTEIN_TYPES,ALL_TYPES

def main(input_gff_path,output_gff_path,valid_chroms):
    gff = read_gff(input_gff_path,valid_features=ALL_TYPES)
    gff = gff[gff['chr'].isin(valid_chroms)]
    gff.loc[:, 'chr'] = gff['chr'].str.replace('Chr', '')
    gff = get_gff_with_attribute(gff, ['parent'])
    gff = dupliacte_gff_by_parent(gff)
    gff = get_gff_with_belonging(gff,gene_types=GENE_TYPES,transcript_types=TRANSCRIPT_TYPES)
    discarded_gene_ids = list(gff[gff['feature'].isin(PRIMARY_MIRNA_TPYES)]['parent'])
    discarded_ids = list(gff[gff['feature'].isin(UORF_TYPES + PROTEIN_TYPES + MIRNA_TPYES)]['id'])
    gff = gff[(~gff['belong_gene'].isin(discarded_gene_ids)) & (~gff['id'].isin(discarded_ids))]
    convert_feature = {}
    input_types = [GENE_TYPES,TRANSCRIPT_TYPES,EXON_TYPES,CDS_TYPES,FIVE_PRIME_UTR_TYPES,THREE_PRIME_UTR_TYPES]
    convert_types = [GENE_TYPE,TRANSCRIPT_TYPE,EXON_TYPE,CDS_TYPE,UTR_TYPE,UTR_TYPE]
    for input_type,convert_type in zip(input_types,convert_types):
        convert_feature.update(dict(zip(input_type,[convert_type]*len(input_type))))
    gff['feature'] = gff['feature'].replace(convert_feature)
    gff = get_gff_with_updated_attribute(gff)
    write_gff(gff, output_gff_path)


if __name__ == "__main__":
    # Reading arguments
    parser = ArgumentParser(
        description='Remove miRNA, miRNA_primary_transcript, uORF and protein '
        'related data, rename chromomsome id and dupliacte gff item by their parent id'
    )
    parser.add_argument("-i", "--input_gff_path", required=True)
    parser.add_argument("-o", "--output_gff_path", required=True)
    parser.add_argument("-v","--valid_chroms",required=True,type=lambda x: x.split(','))
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
