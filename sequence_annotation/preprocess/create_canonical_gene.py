import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_bed, get_gff_with_attribute, read_gff,write_gff, read_bed
from sequence_annotation.preprocess.get_id_table import read_id_table
from sequence_annotation.preprocess.gff2bed import gff_info2bed_info,gff2bed
from sequence_annotation.preprocess.bed2gff import bed2gff
from sequence_annotation.preprocess.utils import EXON_TYPES,RNA_TYPES
from sequence_annotation.preprocess.get_id_table import get_id_table,convert_id_table_to_dict,write_id_table
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import convert_from_gff_to_gene_with_alt_status_gff
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import convert_from_bed_to_gene_with_alt_status_gff

def create_gene_bed_from_exon_gff(gff):
    gff = get_gff_with_attribute(gff)
    transcripts = gff[gff['feature'].isin(RNA_TYPES)]
    ids = set(transcripts['id'])
    transcripts = transcripts.groupby('id')
    exon_types = list(EXON_TYPES)
    exons = gff[gff['feature'].isin(exon_types)]
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
    id_table = get_id_table(get_gff_with_attribute(gene_with_alt_status_gff))
    id_convert_dict = convert_id_table_to_dict(id_table)
    bed = create_gene_bed_from_exon_gff(gene_with_alt_status_gff)
    gff = bed2gff(bed,id_convert_dict)
    return gff


def main(input_path,output_status_path,
         output_gff_path,output_bed_path,
         output_id_table_path,id_table_path=None,**kwargs):
    if 'bed' in input_path.split('.')[-1]:
        if id_table_path is None:
            raise Exception("If input data is bed format, then the id_table_path must be provided")
        bed = read_bed(input_path)
        id_table = read_id_table(id_table_path)
        status_gff = convert_from_bed_to_gene_with_alt_status_gff(bed,id_table,**kwargs)
    else:
        gff = read_gff(input_path)
        status_gff = convert_from_gff_to_gene_with_alt_status_gff(gff,**kwargs)
    gene_gff = create_gene_gff_from_gene_alt_status_gff(status_gff)
    gene_bed = gff2bed(gene_gff)
    gene_id_table = get_id_table(gene_gff)
    
    write_gff(status_gff,output_status_path)
    write_gff(gene_gff,output_gff_path)
    write_bed(gene_bed,args.output_bed_path)
    write_id_table(gene_id_table,output_id_table_path)
    
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-s", "--output_status_path",required=True)
    parser.add_argument("-g", "--output_gff_path",required=True)
    parser.add_argument("-b", "--output_bed_path",required=True)
    parser.add_argument("-o", "--output_id_table_path",required=True)
    parser.add_argument("-t", "--id_table_path")
    parser.add_argument("--select_site_by_election",action='store_true')
    parser.add_argument("--allow_partial_gene",action='store_true')
    args = parser.parse_args()
    main(**vars(args))
