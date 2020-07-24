import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed, write_bed


def get_UTR(bed):
    UTRs = []
    for item in bed.to_dict('record'):
        strand = item['strand']
        if strand not in ['+', '-']:
            raise Exception("Wrong strnad {}".format(strand))
        exon_starts = []
        exon_ends = []
        thick_start = int(item['thick_start'])
        thick_end = int(item['thick_end'])
        start = int(item['start'])
        end = int(item['end'])
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
            
        UTR = {}
        UTR['id'] = item['id']
        UTR['strand'] = strand
        UTR['chr'] = item['chr']
        #Assume all strand is plus strand
        #if transcript is coding:
        if thick_start <= thick_end:
            for exon_start, exon_end in zip(exon_starts,exon_ends):
                #If it is CDS
                if thick_start <= exon_start and exon_end <= thick_end:
                    continue
                UTR_ = dict(UTR)
                UTR_['start'] = exon_start
                UTR_['end'] = exon_end
                #If it is 5' UTR
                if exon_start < thick_start:
                    #If it is 5' UTR that is neighboring by CDS               
                    if exon_end >= thick_start:
                        UTR_['end'] = thick_start-1
                        #If it is external 5' UTR
                    if exon_start == start:
                        UTR_['type'] = 'five_external_utr'
                    else:
                        UTR_['type'] = 'five_internal_utr'
                else:
                    #If it is 3' UTR that is neighboring by CDS
                    if thick_end > exon_start:
                        UTR_['start'] = thick_end+1
                    if exon_end == end:
                        UTR_['type'] = 'three_external_utr'
                    else:
                        UTR_['type'] = 'three_internal_utr'
                UTRs.append(UTR_)

        else:
            for exon_start, exon_end in zip(exon_starts,exon_ends):
                UTR_ = dict(UTR)
                UTR_['start'] = exon_start
                UTR_['end'] = exon_end
                if exon_start == start:
                    UTR_['type'] = 'five_external_utr'
                else:
                    UTR_['type'] = 'five_internal_utr'
                UTRs.append(UTR_)
                    
            for exon_start, exon_end in zip(exon_starts,exon_ends):
                UTR_ = dict(UTR)
                UTR_['start'] = exon_start
                UTR_['end'] = exon_end
                if exon_end == end:
                    UTR_['type'] = 'three_external_utr'
                else:
                    UTR_['type'] = 'three_internal_utr'
                UTRs.append(UTR_)
                
    for UTR_ in UTRs:
        #Set by correct type of strand
        if UTR_['strand'] == '-':
            if 'five' in UTR_['type']:
                UTR_['type'] = UTR_['type'].replace('five','three')
            else:
                UTR_['type'] = UTR_['type'].replace('three','five')
                
    utr_bed = pd.DataFrame.from_dict(UTRs)
    utr_bed['score'] = '.'
    utr_bed['id'] = utr_bed['type']
    utr_bed = utr_bed[['chr', 'start', 'end', 'id', 'score', 'strand']]
    return utr_bed


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path", help="bed_path", required=True)
    parser.add_argument("-o", "--output_path", help="saved_root", required=True)
    args = parser.parse_args()
    bed = read_bed(args.input_path)
    utr_bed = get_UTR(bed)
    write_bed(utr_bed, args.output_path)
