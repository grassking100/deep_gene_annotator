import sys
import os
sys.path.append(os.path.dirname(__file__) + "/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import write_bed, read_bed, read_fasta


def main(bed_path, fasta_path, cleaned_bed_path, dirty_bed_path, codes):

    codes = set(list(codes.upper()))

    bed = read_bed(bed_path)
    fasta = read_fasta(fasta_path, check_unique_id=False)
    if set(bed['id']) != set(fasta.keys()):
        raise Exception()

    cleaned_ids = []
    dirty_ids = []
    for name, seq in fasta.items():
        if len(set(list(seq.upper())) - codes) > 0:
            dirty_ids.append(name)
        else:
            cleaned_ids.append(name)

    cleaned_bed = bed[bed['id'].isin(cleaned_ids)]
    dirty_bed = bed[bed['id'].isin(dirty_ids)]

    write_bed(cleaned_bed, cleaned_bed_path)
    write_bed(dirty_bed, dirty_bed_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-b", "--bed_path", required=True)
    parser.add_argument("-f", "--fasta_path", required=True)
    parser.add_argument("-o", "--cleaned_bed_path", required=True)
    parser.add_argument("-d", "--dirty_bed_path", required=True)
    parser.add_argument("-c", "--codes", default='ATCG')
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
