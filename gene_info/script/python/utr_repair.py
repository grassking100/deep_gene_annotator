import pandas as pd
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))
from utils import get_df_with_seq_id

def repair_exon(group,return_repaired_name=False):
    info_ = dict(group.iloc[0,:].to_dict())
    info_[2] = 'exon'
    list_group = group.to_dict('list')
    starts = list_group[3]
    ends = list_group[4]
    sizes = [end-start+1 for start,end in zip(starts,ends)]
    sum_size = sum(sizes)
    exons_info = []
    indice = np.argsort(starts)
    exon_start = None
    exon_end = None
    update = False
    repaired_name = None
    for index in indice:
        start = starts[index]
        end = ends[index]
        if exon_start is None:
            exon_start = start
        else:
            continue_ = (start == exon_end+1)
            if not continue_:
                info = dict(info_)
                info[3] = exon_start
                info[4] = exon_end
                exons_info.append(info)
                exon_start = start
            else:
                repaired_name = list(group['parent'])[0]
        exon_end = end
    info = dict(info_)
    info[3] = exon_start
    info[4] = exon_end
    exons_info.append(info)
    if sum_size != sum([item[4]-item[3]+1 for item in exons_info]):
        raise Exception('Repair is fail for unkwnown reason')
    if return_repaired_name:
        return exons_info,repaired_name
    else:
        return exons_info

def gff_repair(df,return_repaired_name=False):
    exon_subtype = ['five_prime_UTR','three_prime_UTR','CDS']
    df = get_df_with_seq_id(df)
    gene_ids = set(df[df[2]=='mRNA']['id'])
    is_exon_subtype = df[2].isin(exon_subtype)
    is_exon = df[2] == 'exon'
    exon_subtype_df = df[is_exon_subtype]
    exon_subtype_df = exon_subtype_df.groupby('parent')
    others_df = df[~is_exon]
    exon_df = df[is_exon]
    exon_df = exon_df.groupby('parent')
    exons_info = []
    repaired_names = []
    for name in gene_ids:
        try:
            group = exon_subtype_df.get_group(name)
            exon_info,repaired_name = repair_exon(group,return_repaired_name)
            if repaired_name is not None:
                repaired_names.append(repaired_name)
        except KeyError:
            exon_info = exon_df.get_group(name).to_dict('record')
        exons_info += exon_info
    exons = pd.DataFrame.from_dict(exons_info)
    df = pd.concat([exons,others_df],sort=True)
    df = df.iloc[:,:9]
    if return_repaired_name:
        return df,repaired_names
    else:
        return df
    
if __name__ =='__main__':
    raw_gff_path=sys.argv[1]
    rapiared_gff_path=sys.argv[2]
    repaired_name_path=sys.argv[3]
    df_ = pd.read_csv(raw_gff_path,sep='\t',header=None,comment ='#')
    df,names = gff_repair(df_,return_repaired_name=True)
    df.to_csv(rapiared_gff_path,header=None,index=None,sep='\t')
    with open(repaired_name_path,"w") as fp:
        for name in names:
            fp.write(name+"\n")
    
    