import os, sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import read_bed, write_bed
from sequence_annotation.file_process.get_subbed import get_subbed
from sequence_annotation.file_process.seq_info_parser import BedInfoParser
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import get_most_start_end_transcripts
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import get_canonical_region_and_alt_splice


def get_no_alt_site_transcript(bed,id_convert_dict,select_boundary_by_election=False):
    """Return site-based data"""
    #Read bed file form path
    parser = BedInfoParser()
    mRNAs = parser.parse(bed)
    parents = id_convert_dict
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
            if select_boundary_by_election:
                mRNAs = get_most_start_end_transcripts(mRNAs)
            if len(mRNAs) > 0:
                _, _, alt_ds, alt_as, d_a_site, _ = get_canonical_region_and_alt_splice(mRNAs)
                if len(list(alt_ds) + list(alt_as) + list(d_a_site)) == 0:
                    ids = [mRNA['id'] for mRNA in mRNAs]
                    no_alt_site_transcript_ids += ids
            else:
                print("Cannot get {}'s canonical gene model".format(gene_id))
        except:
            raise Exception("The gene {} causes error".format(gene_id))

    return get_subbed(bed, no_alt_site_transcript_ids)

def main(input_bed_path,id_table_path,output_bed_path,**kwargs):
    bed = read_bed(input_bed_path)
    id_convert_dict = get_id_convert_dict(id_table_path)
    part_bed = get_no_alt_site_transcript(bed,id_convert_dict,**kwargs)

    write_bed(part_bed, output_bed_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_bed_path", required=True)
    parser.add_argument("-t", "--id_table_path", required=True)
    parser.add_argument("-o", "--output_bed_path", required=True)
    parser.add_argument("--select_boundary_by_election",
                        action='store_true')
    args = parser.parse_args()
    main(**vars(args))
