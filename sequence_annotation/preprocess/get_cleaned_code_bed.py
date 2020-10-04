import sys
import os
sys.path.append(os.path.dirname(__file__) + "/../..")
from argparse import ArgumentParser
from sequence_annotation.file_process.utils import write_bed, read_bed, read_fasta

def get_cleaned_code_bed(bed, fasta, codes=None):
    codes = codes or 'ATCG'
    codes = set(list(codes.upper()))
    unexpeet_ids = set(fasta.keys()) - set(bed['id'])
    if unexpeet_ids:
        raise Exception("BED ids are not same as fasta ids, got {}".format(unexpeet_ids))
    cleaned_ids = []
    dirty_ids = []
    for name, seq in fasta.items():
        if len(set(list(seq.upper())) - codes) > 0:
            dirty_ids.append(name)
        else:
            cleaned_ids.append(name)
    cleaned_bed = bed[bed['id'].isin(cleaned_ids)]
    dirty_bed = bed[bed['id'].isin(dirty_ids)]
    return cleaned_bed,dirty_bed


def main(bed_path, fasta_path,cleaned_bed_path, dirty_bed_path=None,codes=None):
    bed = read_bed(bed_path)
    fasta = read_fasta(fasta_path)
    cleaned_bed, dirty_bed = get_cleaned_code_bed(bed, fasta, codes=codes)
    write_bed(cleaned_bed, cleaned_bed_path)
    if dirty_bed_path is not None:
        write_bed(dirty_bed, dirty_bed_path)

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-b", "--bed_path", required=True)
    parser.add_argument("-f", "--fasta_path", required=True)
    parser.add_argument("-o", "--cleaned_bed_path", required=True)
    parser.add_argument("-d", "--dirty_bed_path")
    parser.add_argument("-c", "--codes")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
