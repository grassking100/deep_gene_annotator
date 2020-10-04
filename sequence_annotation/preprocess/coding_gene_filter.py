import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.file_process.utils import write_bed, read_bed,BED_COLUMNS
from sequence_annotation.file_process.get_id_table import get_id_convert_dict


def get_real_protein_transcript_id(transcript_ids,id_convert_dict,real_protein_gene_ids):
    ids = []
    for transcript_id in transcript_ids:
        if id_convert_dict[transcript_id] in real_protein_gene_ids:
            ids.append(transcript_id)
    return ids


def coding_transcript_filter(bed):
    coding_transcript = bed[bed['thick_end']-bed['thick_start']+1>0].copy()
    return coding_transcript


def coding_gene_filter(bed,id_convert_dict):
    bed = bed.copy()
    bed['thick_length'] = bed['thick_end'] - bed['thick_start'] + 1
    has_noncoding_gene_id = set()
    for item in bed.to_dict('record'):
        if item['thick_length'] < 0:
            raise Exception("Negative thick length")
        elif item['thick_length'] == 0:
            has_noncoding_gene_id.add(id_convert_dict[item['id']])

    coding_gene_transcripts = []
    for item in bed.to_dict('record'):
        gene_id = id_convert_dict[item['id']]
        if gene_id not in has_noncoding_gene_id:
            coding_gene_transcripts.append(item)
    if len(coding_gene_transcripts) > 0:
        coding_gene_transcripts = pd.DataFrame.from_dict(coding_gene_transcripts)
    else:
        coding_gene_transcripts = pd.DataFrame(columns=BED_COLUMNS)
    return coding_gene_transcripts
    
    
def main(input_bed_path,id_table_path,output_bed_path):
    bed = read_bed(input_bed_path)
    id_convert_dict = get_id_convert_dict(id_table_path)
    coding_gene_transcripts = coding_gene_filter(bed,id_convert_dict)
    write_bed(coding_gene_transcripts, output_bed_path)
    
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will output trasncripts which their genes have only coding transcripts")
    parser.add_argument("-i","--input_bed_path",help="Input BED file",required=True)
    parser.add_argument("-t","--id_table_path",help="Table about id conversion",required=True)
    parser.add_argument("-o","--output_bed_path",help="Output BED file",required=True)
    args = parser.parse_args()
    main(**vars(args))

