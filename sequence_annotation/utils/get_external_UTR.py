import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed


def get_external_UTR(bed):
    external_UTR = []
    for item in bed.to_dict('record'):
        strand = item['strand']
        if strand not in ['+', '-']:
            raise Exception("Wrong strnad {}".format(strand))
        exon_starts = []
        exon_ends = []
        thick_start = int(item['thick_start'])
        thick_end = int(item['thick_end'])
        start = int(item['start'])
        count = int(item['count'])
        exon_sizes = [
            int(val) for val in item['block_size'].split(',')[:count]
        ]
        exon_rel_starts = [
            int(val) for val in item['block_related_start'].split(',')[:count]
        ]
        for exon_rel_start, exon_size in zip(exon_rel_starts, exon_sizes):
            exon_start = exon_rel_start + start
            exon_end = exon_start + exon_size - 1
            exon_starts.append(exon_start)
            exon_ends.append(exon_end)
        left_target_start = min(exon_starts)
        left_target_end = exon_ends[exon_starts.index(left_target_start)]
        right_target_start = max(exon_starts)
        right_target_end = exon_ends[exon_starts.index(right_target_start)]

        left = {}
        right = {}
        right['id'] = left['id'] = item['id']
        right['strand'] = left['strand'] = strand
        right['chr'] = left['chr'] = item['chr']
        left.update({'start': left_target_start, 'end': left_target_end})
        right.update({'start': right_target_start, 'end': right_target_end})
        #if transcript is coding:
        if thick_start <= thick_end:
            left['end'] = min(left_target_end, thick_start - 1)
            right['start'] = max(right_target_start, thick_end + 1)

        if strand == '+':
            left['type'] = 'five_external_utr'
            right['type'] = 'three_external_utr'
        else:
            left['type'] = 'three_external_utr'
            right['type'] = 'five_external_utr'

        #If left UTR are exist
        if left['start'] <= left['end']:
            external_UTR.append(left)
        #If right UTR are exist
        if right['start'] <= right['end']:
            external_UTR.append(right)

    external_UTR = pd.DataFrame.from_dict(external_UTR)
    return external_UTR


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b", "--bed_path", help="bed_path", required=True)
    parser.add_argument("-s", "--saved_root", help="saved_root", required=True)
    args = parser.parse_args()
    five_UTR_bed_path = os.path.join(args.saved_root, "external_five_UTR.bed")
    three_UTR_bed_path = os.path.join(args.saved_root,
                                      "external_three_UTR.bed")

    if os.path.exists(five_UTR_bed_path) and os.path.exists(
            three_UTR_bed_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        bed = read_bed(args.bed_path)
        utr_bed = get_external_UTR(bed)
        utr_bed['score'] = '.'
        five_external_utr = utr_bed[utr_bed['type'].isin(['five_external_utr'
                                                          ])]
        three_external_utr = utr_bed[utr_bed['type'].isin(
            ['three_external_utr'])]
        five_external_utr_bed = five_external_utr[[
            'chr', 'start', 'end', 'id', 'score', 'strand'
        ]]
        three_external_utr_bed = three_external_utr[[
            'chr', 'start', 'end', 'id', 'score', 'strand'
        ]]
        write_bed(five_external_utr_bed, five_UTR_bed_path)
        write_bed(three_external_utr_bed, three_UTR_bed_path)
