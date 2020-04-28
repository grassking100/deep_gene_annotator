import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import print_progress


def flip_and_rename_gff(gff,region_table):
    """
    The method would flip data's coordinate to its origin strands
    """
    redefined_gff = []
    region_info_dict = {}
    for region in region_table.to_dict('record'):
        region_info_dict[region['ordinal_id_with_strand']] = region
    for index,gff_item in enumerate(gff.to_dict('record')):
        print_progress("{}% of data have been flipped and renamed".format(int(100 * index / len(gff))))
        chrom = gff_item['chr']
        region = region_info_dict[chrom]
        if region['strand'] == '-':
            gff_item['strand'] = '-'
            anchor = region['length'] + 1
            # Start recoordinate
            start =  gff_item['start']
            end =  gff_item['end']
            gff_item['end'] = anchor - start
            gff_item['start'] = anchor - end
            if gff_item['end'] <= 0 or gff_item['start'] <= 0:
                raise Exception("Invalid start or end in {}".format(chrom))
            if gff_item['end'] - gff_item['start'] + 1 <= 0:
                raise Exception("Wrong block size")
        gff_item['chr'] = region['ordinal_id_wo_strand']
        new_id_prefix = gff_item['chr']
        if region['strand'] == '+':
            new_id_prefix += '_plus'
        else:
            new_id_prefix += '_minus'
        gff_item['attribute'] = gff_item['attribute'].replace(
            chrom, new_id_prefix)
        redefined_gff.append(gff_item)

    redefined_gff = pd.DataFrame.from_dict(redefined_gff)
    redefined_gff = redefined_gff.sort_values(by=['chr','start','end','strand'])
    return redefined_gff
