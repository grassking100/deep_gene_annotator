import sys
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/..")
from sequence_annotation.utils.utils import get_gff_with_attribute, read_gff, write_bed


def gff_info2bed_info(cDNAs):
    # input one based data, return one based data
    id_ = set(cDNAs['name'])
    if len(id_) == 1:
        id_ = list(id_)[0]
    else:
        raise Exception("Multiple ids in cDNA cluster, get {}".format(id_))
    starts = list(cDNAs['start'])
    ends = list(cDNAs['end'])
    min_start = min(starts)
    max_end = max(ends)
    indice = np.argsort(starts)
    starts = [starts[index] for index in indice]
    ends = [ends[index] for index in indice]
    sizes = [str(end - start + 1) for start, end in zip(starts, ends)]
    for size in sizes:
        if int(size) <= 0:
            raise Exception(
                "The cDNA size in {} should be positive".format(id_))
    rel_starts = [str(start - min_start) for start in starts]
    info = dict(cDNAs.iloc[0])
    info['id'] = id_
    info['start'] = min_start
    info['end'] = max_end
    info['rgb'] = '.'
    info['score'] = '.'
    info['count'] = len(sizes)
    info['block_size'] = ",".join(sizes)
    info['block_related_start'] = ",".join(rel_starts)
    info['thick_start'] = min_start
    info['thick_end'] = min_start - 1
    return info


def cluster_gmap_cdna_to_bed(gff):
    gff = get_gff_with_attribute(gff)
    cDNA_group = gff[gff['feature'] == 'cDNA_match'].groupby('name')
    bed_info_list = []
    for _, cdnas in cDNA_group:
        bed_info = gff_info2bed_info(cdnas)
        bed_info_list.append(bed_info)
    bed = pd.DataFrame.from_dict(bed_info_list)
    return bed


def main(gff_path, bed_path):
    gff = read_gff(gff_path)
    bed = cluster_gmap_cdna_to_bed(gff)
    write_bed(bed, bed_path)


if __name__ == '__main__':
    parser = ArgumentParser(
        description="This program will group cDNAs with same id together"
        ", and write to bed format")
    parser.add_argument("-i",
                        "--gff_path",
                        required=True,
                        help="Path of input gff file")
    parser.add_argument("-o",
                        "--bed_path",
                        required=True,
                        help="Path of output bed file")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
