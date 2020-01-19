import sys
import os
sys.path.append(os.path.dirname(__file__)+"/../..")
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed

if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("-f","--first_bed_path",required=True)
    parser.add_argument("-s","--second_bed_path",required=True)
    parser.add_argument("-o", "--output_path",required=True)
    args = parser.parse_args()
    first_bed = read_bed(args.first_bed_path)
    second_bed = read_bed(args.second_bed_path)
    first_keys = list(first_bed['id'])
    second_keys = list(second_bed['id'])
    first_bed = first_bed.set_index('id').to_dict('index')
    second_bed = second_bed.set_index('id').to_dict('index')

    if set(first_keys) != set(second_keys) or len(first_keys) != len(second_keys):
        raise Exception("Two bed must have same seqeunces' ids and data should have same number of sequnece")
    keys = []    
    for key,first_item in first_bed.items():
        second_item = second_bed[key]
        start_eq = first_item['thick_start'] == second_item['thick_start']
        end_eq = first_item['thick_end'] == second_item['thick_end']
        if start_eq and end_eq:
            keys.append(key)

    with open(args.output_path,"w") as fp:
        for key in keys:
            fp.write("{}\n".format(key))
