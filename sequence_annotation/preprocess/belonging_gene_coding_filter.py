import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import write_bed, read_bed
from sequence_annotation.preprocess.get_id_table import get_id_convert_dict

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser(
        description=
        "This program will output trasncripts which their genes have only coding transcripts"
    )
    parser.add_argument("-i",
                        "--input_bed_path",
                        help="Input BED file",
                        required=True)
    parser.add_argument("-t",
                        "--id_table_path",
                        help="Table about id conversion",
                        required=True)
    parser.add_argument("-o",
                        "--output_bed_path",
                        help="Output BED file",
                        required=True)
    args = parser.parse_args()

    bed = read_bed(args.input_bed_path)
    try:
        id_convert_dict = get_id_convert_dict(args.id_table_path)
    except:
        raise Exception("Something wrong happened in {}".format(
            args.id_table_path))
    bed['thick_length'] = bed['thick_end'] - bed['thick_start'] + 1

    gene_id_has_noncoding_transcript = set()
    for item in bed.to_dict('record'):
        if item['thick_length'] < 0:
            raise Exception("Negative thick length")
        elif item['thick_length'] == 0:
            gene_id_has_noncoding_transcript.add(id_convert_dict[item['id']])

    all_coding_transcript_gene_transcripts = []
    for item in bed.to_dict('record'):
        gene_id = id_convert_dict[item['id']]
        if gene_id not in gene_id_has_noncoding_transcript:
            all_coding_transcript_gene_transcripts.append(item)
    all_coding_transcript_gene_transcripts = pd.DataFrame.from_dict(
        all_coding_transcript_gene_transcripts)
    write_bed(all_coding_transcript_gene_transcripts, args.output_bed_path)
