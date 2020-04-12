import os
import sys
from Bio import SeqIO
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/..")
from sequence_annotation.utils.utils import read_gff, get_gff_with_attribute, write_gff
from sequence_annotation.preprocess.get_id_table import get_id_table, convert_id_table_to_dict


if __name__ == '__main__':
    parser = ArgumentParser(description="This program will export complete "
                            "ORF GFF and fasta")
    parser.add_argument("--input_fasta_path",
                        required=True,
                        help="Path of input fasta file")
    parser.add_argument("--input_gff_path",
                        required=True,
                        help="Path of input gff file")
    parser.add_argument("--output_fasta_path",
                        required=True,
                        help="Path of output fasta file")
    parser.add_argument("--output_gff_path",
                        required=True,
                        help="Path of output gff file")

    args = parser.parse_args()
    fasta_sequences = SeqIO.parse(open(args.input_fasta_path), 'fasta')
    gff = get_gff_with_attribute(read_gff(args.input_gff_path))
    mRNAs = gff[gff['feature'] == 'mRNA']
    id_table = convert_id_table_to_dict(get_id_table(mRNAs))
    valid_fasta = []
    valid_ids = []
    valid_status = "ORF type:complete"
    for fasta in fasta_sequences:
        if valid_status in fasta.description and fasta.id in id_table:
            valid_ids.append(fasta.id)
            valid_ids.append(id_table[fasta.id])
            valid_fasta.append(fasta)

    output_gff = gff[(gff['id'].isin(valid_ids)) |
                     (gff['parent'].isin(valid_ids))]
    SeqIO.write(valid_fasta, args.output_fasta_path, "fasta")
    write_gff(output_gff, args.output_gff_path)
