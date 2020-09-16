import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.file_process.utils import read_bed, write_bed
from sequence_annotation.file_process.get_id_table import convert_id_table_to_dict, read_id_table
from sequence_annotation.file_process.gff2bed import gff2bed
from sequence_annotation.file_process.utils import RNA_TYPES
from sequence_annotation.file_process.bed2gff import bed2gff
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import get_most_start_end_transcripts,get_cluster_mRNA

def get_coding_gff(gff):
    cds = gff[gff['feature']=='CDS'].copy()
    cds['feature'] = 'exon'
    parents = set(cds['parent'])
    transcripts = gff[(gff['feature'].isin(RNA_TYPES))&(gff['id'].isin(parents))].copy()
    gff = pd.concat([cds,transcripts])
    return gff

def main(input_path,output_bed_path,id_table_path):
    bed = read_bed(input_path)
    id_table = read_id_table(id_table_path)
    id_convert_dict = convert_id_table_to_dict(id_table)
    gff = bed2gff(bed,id_convert_dict)
    gff = get_coding_gff(gff)
    transcript_gff = gff[gff['feature'].isin(RNA_TYPES)]
    transcript_bed = gff2bed(transcript_gff)
    """Return site-based data"""
    genes = get_cluster_mRNA(transcript_bed,id_table)
    #Handle each cluster
    selected_ids = []
    for gene_id,mRNAs in genes.items():
        mRNAs = get_most_start_end_transcripts(mRNAs)
        selected_ids += [mRNA['id'] for mRNA in mRNAs]

    selected_gff = gff[(gff['parent'].isin(selected_ids)) | (gff['id'].isin(selected_ids))]
    selected_bed = gff2bed(selected_gff)
    selected_bed['id'] = selected_bed['id'].map(id_convert_dict)
    selected_bed = selected_bed.drop_duplicates()
    selected_bed['coord_id'] = selected_bed['chr']+selected_bed['strand']
    selected_bed['coord_id'] = selected_bed['coord_id'] + selected_bed['start'].astype(str)+selected_bed['end'].astype(str)
    selected_bed['coord_id'] = selected_bed['coord_id'] +selected_bed['block_starts']+selected_bed['block_sizes']
    selected_groups = selected_bed.groupby('id')
    
    for id_ in selected_groups.groups:
        group = selected_groups.get_group(id_)
        if len(set(group['coord_id']))!=1:
            raise Exception(id_)
        
    write_bed(selected_bed,args.output_bed_path)
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-o", "--output_bed_path",required=True)
    parser.add_argument("-t", "--id_table_path",required=True)
    args = parser.parse_args()
    main(**vars(args))
