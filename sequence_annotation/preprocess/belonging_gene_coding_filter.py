import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.file_process.utils import write_bed, read_bed
from sequence_annotation.file_process.get_id_table import get_id_convert_dict

def belonging_gene_coding_filter(bed,id_convert_dict):
    bed = bed.copy()
    bed['thick_length'] = bed['thick_end'] - bed['thick_start'] + 1
    gene_id_has_noncoding_transcript = set()
    for item in bed.to_dict('record'):
        if item['thick_length'] < 0:
            raise Exception("Negative thick length")
        elif item['thick_length'] == 0:
            gene_id_has_noncoding_transcript.add(id_convert_dict[item['id']])

    coding_gene_transcript_id = []
    for item in bed.to_dict('record'):
        gene_id = id_convert_dict[item['id']]
        if gene_id not in gene_id_has_noncoding_transcript:
            coding_gene_transcript_id.append(item)
    coding_gene_transcript_id = pd.DataFrame.from_dict(coding_gene_transcript_id)
    return coding_gene_transcript_id
    
    
def main(input_bed_path,id_table_path,output_bed_path):
    bed = read_bed(input_bed_path)
    id_convert_dict = get_id_convert_dict(id_table_path)
    coding_gene_transcript_id = belonging_gene_coding_filter(bed,id_convert_dict)
    write_bed(coding_gene_transcript_id, output_bed_path)
    
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(description="This program will output trasncripts which their genes have only coding transcripts")
    parser.add_argument("-i","--input_bed_path",help="Input BED file",required=True)
    parser.add_argument("-t","--id_table_path",help="Table about id conversion",required=True)
    parser.add_argument("-o","--output_bed_path",help="Output BED file",required=True)
    args = parser.parse_args()
    main(**vars(args))

