import sys
import os
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_bed

def main(bed_path, output_path):
    bed = read_bed(bed_path)
    bed['coding_length'] = bed['thick_end'] - bed['thick_start'] + 1
    ids = set(bed[bed['coding_length']>0]['id'])
    with open(output_path,'w') as fp:
        for id_ in ids:
            fp.write("{}\n".format(id_))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--bed_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
