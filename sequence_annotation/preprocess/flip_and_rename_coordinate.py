import pandas as pd

def _create_fai_by_region_table(region_table,chrom_key=None):
    chrom_key = chrom_key or 'old_id'
    ids = list(region_table[chrom_key])
    lengths = list(region_table['end']-region_table['start']+1)
    fai = dict(zip(ids,lengths))
    return fai

def flip_and_rename_bed(bed,region_table,chrom_source=None,chrom_target=None):
    """
    The method would flip data's coordinate to its origin strands, and
    get chromosome by chrom_source and rename it by chrom_target 
    """
    chrom_source = chrom_source or 'old_id'
    chrom_target = chrom_target or 'new_id'
    fai = _create_fai_by_region_table(region_table,chrom_source)
    redefined_bed = []
    for item in bed.to_dict('record'):
        bed_item = dict(item)
        chrom = item['chr']
        region = region_table[region_table[chrom_source]==chrom]
        if len(region) != 1:
            raise Exception("Cannot locate for {}".format(chrom))
        region = region.to_dict('record')[0]
        if region['strand'] == '-':
            bed_item['strand'] = '-'
            anchor = fai[chrom] + 1
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
        bed_item['chr'] = region[chrom_target]
        bed_item['id'] = bed_item['id'].replace(chrom,bed_item['chr'])
        redefined_bed.append(bed_item)

    redefined_bed = pd.DataFrame.from_dict(redefined_bed).sort_values(by=['id'])
    return redefined_bed

def flip_and_rename_gff(gff,region_table,chrom_source=None,chrom_target=None):
    """
    The method would flip data's coordinate to its origin strands, and
    get chromosome by chrom_source and rename it by chrom_target 
    """
    chrom_source = chrom_source or 'old_id'
    chrom_target = chrom_target or 'new_id'
    fai = _create_fai_by_region_table(region_table,chrom_source)
    redefined_gff = []
    for item in gff.to_dict('record'):
        gff_item = dict(item)
        chrom = gff_item['chr']
        region = region_table[region_table[chrom_source] == chrom]
        if len(region) != 1:
            raise Exception("Cannot locate for {}".format(chrom))
        region = region.to_dict('record')[0]
        if region['strand'] == '-':
            gff_item['strand'] = '-'
            anchor = fai[chrom] + 1
            #Start recoordinate
            gff_item['end'] = anchor - item['start']
            gff_item['start'] = anchor - item['end']
        gff_item['chr'] = region[chrom_target]
        new_id_prefix = gff_item['chr']
        if region['strand'] == '+':
            new_id_prefix += '_plus'
        else:
            new_id_prefix += '_minus'
        gff_item['attribute'] = gff_item['attribute'].replace(chrom,new_id_prefix)
        redefined_gff.append(gff_item)

    redefined_gff = pd.DataFrame.from_dict(redefined_gff)
    return redefined_gff
