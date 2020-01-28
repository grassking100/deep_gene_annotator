#import os,sys
#sys.path.append(os.path.dirname(__file__)+"/../..")
import pandas as pd
from sequence_annotation.utils.utils import read_bed,write_bed,write_gff,read_gff

def _create_fai_by_region_table(region_table):
    ids = list(region_table['old_id'])
    lengths = list(region_table['end']-region_table['start']+1)
    fai = dict(zip(ids,lengths))
    return fai

def flip_bed(bed,region_table):
    fai = _create_fai_by_region_table(region_table)
    redefined_bed = []
    for item in bed.to_dict('record'):
        bed_item = dict(item)
        old_chrom = item['chr']
        region = region_table[region_table['old_id']==old_chrom]
        if len(region) != 1:
            raise Exception("Cannot locate for {}".format(old_chrom))
        region = region.to_dict('record')[0]
        new_chrom = region['new_id']
        if region['strand'] == '-':
            bed_item['strand'] = '-'
            anchor = fai[old_chrom] + 1
            block_starts = [int(v)+bed_item['start'] for v in  bed_item['block_related_start'].split(",")]
            block_sizes = [v for v in  bed_item['block_size'].split(",")]
            block_ends = [block_start + int(block_size) -1 for block_start,block_size in zip(block_starts,block_sizes)]
            #Start recoordinate
            bed_item['end'] = anchor - item['start']
            bed_item['start'] = anchor - item['end']
            bed_item['thick_end'] = anchor - item['thick_start']
            bed_item['thick_start'] = anchor - item['thick_end']
            new_block_starts = [str(anchor - block_end - bed_item['start']) for block_end in block_ends]
            bed_item['block_related_start'] = ','.join(reversed(new_block_starts))
            bed_item['block_size'] = ','.join(reversed(block_sizes))
        bed_item['chr'] = new_chrom
        bed_item['id'] = bed_item['id'].replace(old_chrom,new_chrom)
        redefined_bed.append(bed_item)

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def flip_gff(gff,region_table):
    fai = _create_fai_by_region_table(region_table)
    redefined_gff = []
    for item in gff.to_dict('record'):
        gff_item = dict(item)
        old_chrom = gff_item['chr']
        region = region_table[region_table['old_id'] == old_chrom]
        if len(region) != 1:
            raise Exception("Cannot locate for {}".format(old_chrom))
        region = region.to_dict('record')[0]
        new_chrom = region['new_id']
        if region['strand'] == '-':
            gff_item['strand'] = '-'
            anchor = fai[old_chrom] + 1
            #Start recoordinate
            gff_item['end'] = anchor - item['start']
            gff_item['start'] = anchor - item['end']
        gff_item['chr'] = new_chrom
        new_id_prefix = new_chrom
        if region['strand'] == '+':
            new_id_prefix += '_plus'
        else:
            new_id_prefix += '_minus'
        gff_item['attribute'] = gff_item['attribute'].replace(old_chrom,new_id_prefix)
        redefined_gff.append(gff_item)

    redefined_gff = pd.DataFrame.from_dict(redefined_gff)
    return redefined_gff
