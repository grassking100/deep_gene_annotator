import sys
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import get_gff_with_attribute, read_gff, write_bed
from sequence_annotation.file_process.utils import GENE_TYPE, TRANSCRIPT_TYPE, CDS_TYPE, EXON_TYPE

def gff_info2bed_info(RNA, exons, orf_info):
    # input one based data, return one based data
    RNA_start = RNA['start']
    exon_starts = list(exons['start'])
    exon_ends = list(exons['end'])
    indice = np.argsort(exon_starts)
    exon_starts = [exon_starts[index] for index in indice]
    exon_ends = [exon_ends[index] for index in indice]
    exon_sizes = [
        str(end - start + 1) for start, end in zip(exon_starts, exon_ends)
    ]
    for size in exon_sizes:
        if int(size) <= 0:
            raise Exception("Exon size in {} should be positive".format(
                RNA['id']))
    exon_rel_starts = [str(start - RNA_start) for start in exon_starts]
    for start in exon_rel_starts:
        if int(start) < 0:
            raise Exception("Exon relative start site shold be nonnegative",
                            RNA, exon_starts, exon_rel_starts)
    info = dict(RNA)
    info['rgb'] = '.'
    info['count'] = len(exon_sizes)
    info['block_size'] = ",".join(exon_sizes)
    info['block_related_start'] = ",".join(exon_rel_starts)
    info['thick_start'] = orf_info['thick_start']
    info['thick_end'] = orf_info['thick_end']
    return info


def extract_orf(selected_CDSs):
    info_ = dict(selected_CDSs.iloc[0, :].to_dict())
    id_ = info_['id']
    thick_start = min(selected_CDSs['start'])
    thick_end = max(selected_CDSs['end'])
    return {'id': id_, 'thick_start': thick_start, 'thick_end': thick_end}


def simple_gff2bed(gff):
    gff = get_gff_with_attribute(gff)
    bed = gff[['chr', 'start', 'end', 'score', 'strand','id']].copy()
    bed['thick_start'] = bed['start']
    bed['thick_end'] = bed['start'] - 1
    bed['rgb'] = '.'
    bed['count'] = 1
    bed['block_size'] = (bed['end'] - bed['start'] + 1).astype(str)
    bed['block_related_start'] = '0'
    return bed


def gff2bed(gff):
    gff = get_gff_with_attribute(gff)
    RNAs = gff[gff['feature']==TRANSCRIPT_TYPE].copy()

    if 'conf_class' in RNAs.columns:
        ratings = []
        for rating in RNAs['conf_class']:
            if rating is not None:
                ratings.append(rating)
            else:
                ratings.append(".")
        RNAs.loc[:, 'score'] = ratings

    if len(RNAs) == 0:
        RNAs = gff[gff['feature']==GENE_TYPE]

    exons = gff[gff['feature']==EXON_TYPE].groupby('parent')
    CDSs = gff[gff['feature']==CDS_TYPE].groupby('parent')
    bed_info_list = []
    for rna in RNAs.to_dict('record'):
        id_ = rna['id']
        if id_ in exons.groups.keys():
            selected_exons = exons.get_group(id_)
        else:
            selected_exons = pd.DataFrame.from_dict([rna])
        if id_ in CDSs.groups.keys():
            selected_CDSs = CDSs.get_group(id_)
            orf = extract_orf(selected_CDSs)
        else:
            orf = {
                'id': id_,
                'thick_start': rna['start'],
                'thick_end': rna['start'] - 1
            }
        bed_info = gff_info2bed_info(rna, selected_exons, orf)
        bed_info_list.append(bed_info)
    bed = pd.DataFrame.from_dict(bed_info_list)
    return bed


def main(gff_path, bed_path, simple_mode=False):
    gff = read_gff(gff_path)
    if simple_mode:
        bed = simple_gff2bed(gff)
    else:
        bed = gff2bed(gff)
    write_bed(bed, bed_path)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="This program will convert gff file to bed file. "
        "It will treat alt_acceptor and alt_donor regions as intron")
    parser.add_argument("-i","--gff_path",required=True,
                        help="Path of input gff file")
    parser.add_argument("-o","--bed_path",required=True,
                        help="Path of output bed file")
    parser.add_argument("-m","--simple_mode",action='store_true',
                        default=False,help="Use simple mode or not")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
