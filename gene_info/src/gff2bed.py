import pandas as pd
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))
from utils import get_df_with_seq_id, BED_COLUMNS

def gff_info2bed_info(mRNA,exon,orf_info):
    mRNA_start = mRNA[3]
    exon_starts = exon[3]
    exon_ends = exon[4]
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
    info = {}
    info['id'] = mRNA['id']
    info['start'] = mRNA_start - 1
    info['end'] = mRNA[4]
    info['strand'] = mRNA[6]
    info['rgb'] = '.'
    info['chr'] = mRNA[0]
    info['count'] = len(exon_sizes)
    info['block_size'] = ",".join(exon_sizes)
    info['block_related_start'] = ",".join(exon_rel_starts)
    info['score'] = '.'
    info['orf_start'] = orf_info['orf_start'] - 1
    info['orf_end'] = orf_info['orf_end']
    return info

def extract_orf(CDS_df,selected_id):
    group = CDS_df.get_group(selected_id)
    info_ = dict(group.iloc[0,:].to_dict())
    id_ = info_['id']
    orf_start = min(group[3])
    orf_end = max(group[4])
    return {'id':id_,'orf_start':orf_start,'orf_end':orf_end}

def gff2bed(gff_path,bed_path):
    gff = pd.read_csv(gff_path,sep='\t',header=None,comment ='#')
    gff = get_df_with_seq_id(gff)
    mRNAs = gff[gff[2]=='mRNA']
    ids = set(mRNAs['id'])
    mRNAs = mRNAs.groupby('id')
    CDSs = gff[gff[2]=='CDS']
    exons = gff[gff[2]=='exon']
    CDSs = CDSs.groupby('parent')
    exons = exons.groupby('parent')
    bed_info_list = []
    for id_ in ids:
        mRNA = mRNAs.get_group(id_).to_dict('record')[0]
        try:
            exon = exons.get_group(id_).to_dict('list')
            try:
                orf = extract_orf(CDSs,id_)
            except KeyError:
                orf = {'id':id_,'orf_start':mRNA[3],'orf_end':mRNA[4]}    
            bed_info = gff_info2bed_info(mRNA,exon,orf)
            bed_info_list.append(bed_info)
        except KeyError:
            pass
    bed = pd.DataFrame.from_dict(bed_info_list)
    bed = bed[BED_COLUMNS]
    bed.to_csv(bed_path,header=None,index=None,sep='\t')
    
if __name__ =='__main__':
    gff_path=sys.argv[1]
    bed_path=sys.argv[2]
    gff2bed(gff_path,bed_path)