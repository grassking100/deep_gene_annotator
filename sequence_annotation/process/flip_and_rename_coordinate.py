import os
import sys
import pandas as pd
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import get_gff_with_updated_attribute,create_empty_gff


def flip_and_rename_gff(gff,region_table):
    """
    The method would flip data's coordinate to its origin strands
    """
    if len(gff)==0:
        return create_empty_gff(gff.columns)
    
    if set(gff['strand']) != set(['+']):
        raise Exception(set(gff['strand']))
    gff = get_gff_with_updated_attribute(gff)
    redefined_gff = []
    region_groups = region_table.groupby('ordinal_id_with_strand')
    for chrom , group in gff.groupby('chr'):
        region = region_groups.get_group(chrom)
        if len(region) != 1:
            raise
        region = region.iloc[0]
        group = group.copy()
        if region['strand'] == '-':
            group['strand'] = '-'
            anchor = region['length'] + 1
            # Start recoordinate
            start =  group['start'].copy()
            end =  group['end'].copy()
            group['end'] = anchor - start
            group['start'] = anchor - end
            if (group['end'] <= 0).any() or (group['start'] <= 0).any():
                raise Exception("Invalid start or end in {}".format(chrom))
            if (group['end'] - group['start'] + 1 <= 0).any():
                raise Exception("Wrong block size")
        new_id_prefix = region['ordinal_id_wo_strand']
        group['chr'] = new_id_prefix
        if region['strand'] == '+':
            new_id_prefix += '_plus'
        else:
            new_id_prefix += '_minus'
        group['attribute'] = group['attribute'].replace(chrom, new_id_prefix)
        redefined_gff.append(group)
    redefined_gff = pd.concat(redefined_gff)
    redefined_gff = redefined_gff.sort_values(by=['chr','start','end','strand'])
    return redefined_gff
