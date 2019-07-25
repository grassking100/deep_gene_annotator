import sys
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.gene_info.utils import GFF_COLUMNS
from sequence_annotation.utils.utils import get_gff_with_seq_id, read_gff, write_bed

def gff_info2bed_info(mRNA,exons,orf_info):
    mRNA_start = mRNA['start']
    exon_starts = exons['start']
    exon_ends = exons['end']
    indice = np.argsort(exon_starts)
    exon_starts = [exon_starts[index] for index in indice]
    exon_ends = [exon_ends[index] for index in indice]
    exon_sizes = [str(end-start+1) for start,end in zip(exon_starts,exon_ends)]
    for size in exon_sizes:
        if int(size) <= 0:
            raise Exception("Exon size shold be positive")
    exon_rel_starts = [str(start-mRNA_start) for start in exon_starts]
    for start in exon_rel_starts:
        if int(start) < 0:
            raise Exception("Exon relative start site shold be nonnegative",mRNA,exon_starts,exon_rel_starts)
    info = dict(mRNA)
    info['start'] -= 1
    info['rgb'] = '.'
    info['count'] = len(exon_sizes)
    info['block_size'] = ",".join(exon_sizes)
    info['block_related_start'] = ",".join(exon_rel_starts)
    info['thick_start'] = orf_info['thick_start'] - 1
    info['thick_end'] = orf_info['thick_end']
    return info

def extract_orf(CDS_df,selected_id):
    group = CDS_df.get_group(selected_id)
    info_ = dict(group.iloc[0,:].to_dict())
    id_ = info_['id']
    thick_start = min(group['start'])
    thick_end = max(group['end'])
    return {'id':id_,'thick_start':thick_start,'thick_end':thick_end}

def simple_gff2bed(gff,bed_path):
    gff = gff.to_dict('record')
    bed_info_list = []
    for item in gff:
        bed_item = {}
        for index,name in enumerate(GFF_COLUMNS):
            bed_item[name] = item[index]    
        bed_item['start'] -= 1
        bed_item['id'] = bed_item['source']
        bed_info_list.append(bed_item)
    bed = pd.DataFrame.from_dict(bed_info_list)
    
def gff2bed(gff,bed_path):
    gff = get_gff_with_seq_id(gff)
    mRNAs = gff[gff['feature']=='mRNA']
    if len(mRNAs) == 0:
        mRNAs = gff[gff['feature']=='gene']
    ids = set(mRNAs['id'])
    mRNAs = mRNAs.groupby('id')
    CDSs = gff[gff['feature']=='CDS']
    exons = gff[gff['feature']=='exon']
    CDSs = CDSs.groupby('parent')
    exons = exons.groupby('parent')
    bed_info_list = []
    for id_ in ids:
        mRNA = mRNAs.get_group(id_).to_dict('record')[0]
        exons_ = exons.get_group(id_).to_dict('list')
        try:
            orf = extract_orf(CDSs,id_)
        except KeyError:
            orf = {'id':id_,'thick_start':mRNA['start'],'thick_end':mRNA['start']-1}
        bed_info = gff_info2bed_info(mRNA,exons_,orf)
        bed_info_list.append(bed_info)
    bed = pd.DataFrame.from_dict(bed_info_list)
    return bed
    
if __name__ =='__main__':
    parser = ArgumentParser(description="This program will convert gff file to bed file")
    parser.add_argument("-i", "--gff_path", help="Path of input gff file",required=True)
    parser.add_argument("-o", "--bed_path", help="Path of output bed file",required=True)
    parser.add_argument("-m", "--simple_mode",type=lambda x: x=='true',
                        default=False,help="Use simple mode or not")
    args = parser.parse_args()
    gff = read_gff(args.gff_path)
    if args.simple_mode:
        bed = simple_gff2bed(gff)
    else:    
        bed = gff2bed(gff)
    write_bed(bed,args.bed_path)    
