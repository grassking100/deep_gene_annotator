import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_bed, get_gff_with_attribute, read_gff
from sequence_annotation.preprocess.gff2bed import gff_info2bed_info
from sequence_annotation.preprocess.utils import EXON_TYPES,RNA_TYPES

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",required=True)
    parser.add_argument("-o", "--output_gene_bed_path",required=True)
    
    args = parser.parse_args()
    gff = read_gff(args.input_path)
    bed = create_gene_bed_from_exon_gff(gff)
    write_bed(bed,args.output_gene_bed_path)
