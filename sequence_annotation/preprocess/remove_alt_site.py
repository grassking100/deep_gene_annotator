import os, sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_bed, write_bed
from sequence_annotation.genome_handler.seq_info_parser import BedInfoParser
from sequence_annotation.preprocess.convert_transcript_to_gene_with_alt_status_gff import get_most_start_end_transcripts
from sequence_annotation.preprocess.convert_transcript_to_gene_with_alt_status_gff import get_canonical_region_and_alt_splice
from sequence_annotation.preprocess.get_subbed import get_subbed


def get_no_alt_site_transcript_id(bed_path,
                                  relation_path,
                                  select_start_end_site_by_election=False):
    """Return site-based data"""
    #Read bed file form path
    parser = BedInfoParser()
    mRNAs = parser.parse(bed_path)
    parents = pd.read_csv(relation_path, sep='\t').to_dict('list')
    parents = dict(zip(parents['transcript_id'], parents['gene_id']))
    #Cluster mRNAs to genes
    genes = {}
    for mRNA in mRNAs:
        parent = parents[mRNA['id']]
        if parent not in genes.keys():
            genes[parent] = []
        genes[parent].append(mRNA)

    #Handle each cluster
    no_alt_site_transcript_ids = []
    for gene_id, mRNAs in genes.items():
        try:
            if select_start_end_site_by_election:
                mRNAs = get_most_start_end_transcripts(mRNAs)
            if len(mRNAs) > 0:
                _, _, alt_ds, alt_as, d_a_site, _ = get_canonical_region_and_alt_splice(
                    mRNAs)
                if len(list(alt_ds) + list(alt_as) + list(d_a_site)) == 0:
                    ids = [mRNA['id'] for mRNA in mRNAs]
                    no_alt_site_transcript_ids += ids
            else:
                print("Cannot get {}'s canonical gene model".format(gene_id))
        except:
            raise Exception("The gene {} causes error".format(gene_id))

    return no_alt_site_transcript_ids


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path", required=True)
    parser.add_argument("-t", "--id_table_path", required=True)
    parser.add_argument("-o", "--output_bed_path", required=True)
    parser.add_argument("--select_start_end_site_by_election",
                        action='store_true')
    args = parser.parse_args()
    ids = get_no_alt_site_transcript_id(args.input_bed_path,
                                        args.id_table_path,
                                        select_start_end_site_by_election=args.
                                        select_start_end_site_by_election)
    bed = read_bed(args.input_bed_path)
    part_bed = get_subbed(bed, ids)
    write_bed(part_bed, args.output_bed_path)
