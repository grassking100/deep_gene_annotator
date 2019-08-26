import os, sys
sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from argparse import ArgumentParser
from sequence_annotation.utils.utils import read_bed,write_bed

def get_external_UTR(bed):
    external_UTR = []
    for item in bed.to_dict('record'):
        exon_starts = []
        exon_ends = []
        thick_start = int(item['thick_start'])
        thick_end = int(item['thick_end'])
        start = int(item['start'])
        count = int(item['count'])
        exon_sizes = [int(val) for val in item['block_size'].split(',')[:count]]
        exon_rel_starts = [int(val) for val in item['block_related_start'].split(',')[:count]]
        for exon_rel_start,exon_size in zip(exon_rel_starts,exon_sizes):
            exon_start = exon_rel_start + start
            exon_end = exon_start + exon_size - 1
            exon_starts.append(exon_start)
            exon_ends.append(exon_end)
        left_target_start = min(exon_starts)
        left_target_end = exon_ends[exon_starts.index(left_target_start)]
        right_target_start = max(exon_starts)
        right_target_end = exon_ends[exon_starts.index(right_target_start)]
        strand = item['strand']
        five = {'start':left_target_start,
                'end':min(left_target_end,thick_start-1),'type':'five_external_utr'}
        three = {'start':max(right_target_start,thick_end+1),
                 'end':right_target_end,'type':'three_external_utr'}
        if strand not in ['+','-']:
            raise Exception("Wrong strnad {}".format(strand))
        if strand == '-':
            five_, three_ = dict(five), dict(three)
            five, three = three_, five_
            three['type'] = 'three_external_utr'
            five['type'] = 'five_external_utr'
        three['id'] = five['id'] = item['id']
        three['strand'] = five['strand'] = strand
        three['chr'] = five['chr'] = item['chr']
        if five['start'] <= five['end']:
            external_UTR.append(five)
        if three['start'] <= three['end']:
            external_UTR.append(three)
    external_UTR = pd.DataFrame.from_dict(external_UTR)
    return external_UTR

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b","--bed_path",
                        help="bed_path",required=True)
    parser.add_argument("-s", "--saved_root",
                        help="saved_root",required=True)
    args = parser.parse_args()

    five_UTR_tsv_path = os.path.join(args.saved_root,"external_five_UTR.tsv")
    three_UTR_tsv_path = os.path.join(args.saved_root,"external_three_UTR.tsv")
    five_UTR_bed_path = os.path.join(args.saved_root,"external_five_UTR.bed")
    three_UTR_bed_path = os.path.join(args.saved_root,"external_three_UTR.bed")

    if os.path.exists(five_UTR_bed_path) and os.path.exists(three_UTR_bed_path):
        print("Result files are already exist, procedure will be skipped.")
    else:
        bed = read_bed(args.bed_path)
        utr_bed = get_external_UTR(bed)
        utr_bed['chr'] = utr_bed['chr'].str.replace('Chr','')
        utr_bed['score']='.'
        five_external_utr = utr_bed[utr_bed['type']=='five_external_utr']
        three_external_utr = utr_bed[utr_bed['type']=='three_external_utr']
        five_external_utr.to_csv(five_UTR_tsv_path,sep='\t',index=None)
        three_external_utr.to_csv(three_UTR_tsv_path,sep='\t',index=None)
        five_external_utr_bed = five_external_utr[['chr','start','end','id','score','strand']].copy()
        three_external_utr_bed = three_external_utr[['chr','start','end','id','score','strand']].copy()
        write_bed(five_external_utr_bed,five_UTR_bed_path)
        write_bed(three_external_utr_bed,three_UTR_bed_path)
