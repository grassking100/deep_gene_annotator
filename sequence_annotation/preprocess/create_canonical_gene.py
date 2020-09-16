import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.file_process.utils import get_gff_with_attribute, read_bed, write_bed, write_gff
from sequence_annotation.file_process.get_id_table import read_id_table
from sequence_annotation.file_process.gff2bed import gff_info2bed_info,gff2bed
from sequence_annotation.file_process.bed2gff import bed2gff
from sequence_annotation.file_process.utils import EXON_TYPE,TRANSCRIPT_TYPE,SUBEXON_TYPES
from sequence_annotation.file_process.utils import ALT_STATUSES,BASIC_GFF_FEATURES
from sequence_annotation.file_process.get_id_table import get_id_table,convert_id_table_to_dict,write_id_table
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import convert_from_bed_to_gene_with_alt_status_gff

def create_gene_bed_from_exon_gff(gff):
    if 'parent' not in gff.columns:
        gff = get_gff_with_attribute(gff)
    transcripts = gff[gff['feature']==TRANSCRIPT_TYPE]
    ids = set(transcripts['id'])
    transcripts = transcripts.groupby('id')
    exons = gff[gff['feature']==EXON_TYPE]
    exons = exons.groupby('parent')
    bed_info_list = []
    for id_ in ids:
        transcript = transcripts.get_group(id_).to_dict('record')[0]
        exons_ = exons.get_group(id_).to_dict('list')
        orf = {'id':id_,'thick_start':transcript['start'],'thick_end':transcript['start']-1}
        bed_info_list.append(gff_info2bed_info(transcript,exons_,orf))
    bed = pd.DataFrame.from_dict(bed_info_list)
    return bed


def create_gene_gff_from_gene_alt_status_gff(gene_with_alt_status_gff):
    id_table = get_id_table(gene_with_alt_status_gff)
    id_convert_dict = convert_id_table_to_dict(id_table)
    bed = create_gene_bed_from_exon_gff(gene_with_alt_status_gff)
    gff = bed2gff(bed,id_convert_dict)
    gff = gff[~gff['feature'].isin(SUBEXON_TYPES)]
    return gff


def main(input_bed_path,id_table_path,
         gene_gff_path=None,output_id_table_path=None,
         status_gff_path=None,gene_bed_path=None,**kwargs):
    bed = read_bed(input_bed_path)
    id_table = read_id_table(id_table_path)
    status_gff = convert_from_bed_to_gene_with_alt_status_gff(bed,id_table,**kwargs)
    gene_gff = create_gene_gff_from_gene_alt_status_gff(status_gff)
    gene_bed = gff2bed(gene_gff)
    gene_id_table = get_id_table(gene_gff)
    
    if gene_gff_path is not None:
        write_gff(gene_gff,gene_gff_path)
    if gene_bed_path is not None:
        write_bed(gene_bed,gene_bed_path)
    if status_gff_path is not None:
        write_gff(status_gff,status_gff_path,valid_features=ALT_STATUSES+BASIC_GFF_FEATURES)
    if output_id_table_path is not None:
        write_id_table(gene_id_table,output_id_table_path)
    
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path",required=True)
    parser.add_argument("-s", "--status_gff_path",required=True)
    parser.add_argument("-g", "--gene_gff_path",required=True)
    parser.add_argument("-b", "--gene_bed_path",required=True)
    parser.add_argument("-o", "--output_id_table_path",required=True)
    parser.add_argument("-t", "--id_table_path",required=True)
    parser.add_argument("--select_site_by_election",action='store_true')
    parser.add_argument("--allow_partial_gene",action='store_true')
    args = parser.parse_args()
    main(**vars(args))
