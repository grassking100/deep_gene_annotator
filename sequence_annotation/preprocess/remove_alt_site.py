import os, sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import read_bed, write_bed
from sequence_annotation.file_process.seq_info_parser import BedInfoParser
from sequence_annotation.preprocess.create_gene_with_alt_status_gff import get_canonical_region_and_alt_splice


def extend_transcript_to_max_boundary(transcripts):
    starts = []
    ends = []
    extended_transcripts = []
    for transcript in transcripts:
        starts.append(transcript['start'])
        ends.append(transcript['end'])
    min_start,max_end = min(starts),max(ends)
    max_extended_size = max_end - min_start + 1
    for transcript in transcripts:
        block_related_starts = []
        block_related_ends = []
        extended_transcript = dict(transcript)
        start_diff = transcript['start'] - min_start
        block_related_starts.append(0)
        for rel_start in transcript['block_related_start'][1:]:
            block_related_starts.append(rel_start+start_diff)
            
        for rel_end in transcript['block_related_end'][:-1]:
            block_related_ends.append(rel_end+start_diff)
            
        block_related_ends.append(max_extended_size-1)
        extended_transcript['block_related_start'] = block_related_starts
        extended_transcript['block_related_end'] = block_related_ends
        extended_transcript['start'] = min_start
        extended_transcript['end'] = max_end
        extended_transcripts.append(extended_transcript)
    return extended_transcripts

def get_no_alt_site_transcript(bed,id_convert_dict):
    """Return site-based data"""
    parser = BedInfoParser()
    transcripts = parser.parse(bed)
    parents = id_convert_dict
    #Cluster transcripts to genes
    genes = {}
    for transcript in transcripts:
        parent = parents[transcript['id']]
        if parent not in genes.keys():
            genes[parent] = []
        genes[parent].append(transcript)

    #Handle each cluster
    no_alt_site_transcript_ids = []
    for gene_id, transcripts in genes.items():
        extended_transcripts = extend_transcript_to_max_boundary(transcripts)
        try:
            _, _, alt_ds, alt_as, d_a_site, _ = get_canonical_region_and_alt_splice(extended_transcripts)
        except:
            raise Exception("The gene {} causes error".format(gene_id))
        if len(list(alt_ds) + list(alt_as) + list(d_a_site)) == 0:
            ids = [transcript['id'] for transcript in transcripts]
            no_alt_site_transcript_ids += ids
    return bed[bed['id'].isin(no_alt_site_transcript_ids)]


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
