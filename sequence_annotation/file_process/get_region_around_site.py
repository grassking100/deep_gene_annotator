import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import TRANSCRIPT_TYPE, INTRON_TYPE


def get_region_around_site(gff, feature_types, upstream_distance,
                           downstream_distance, is_start_location=True):
    blocks = gff[gff['feature'].isin(feature_types)]
    if len(set(blocks['strand']) - set(['+', '-'])) > 0:
        raise Exception("Invalid strand")
    blocks = blocks[['chr', 'strand', 'start', 'end','parent']].copy()
    blocks['transcript_source'] = blocks['parent']
    blocks['site'] = None
    blocks['id'] = ["site_{}".format(i) for i in range(1,1+len(blocks))]
    blocks['score'] = '.'
    plus_index = blocks[(blocks['strand'] == '+')].index
    minus_index = blocks[(blocks['strand'] == '-')].index
    if is_start_location:
        blocks.loc[plus_index, 'site'] = blocks.loc[plus_index, 'start']
        blocks.loc[minus_index, 'site'] = blocks.loc[minus_index, 'end']
    else:
        blocks.loc[plus_index, 'site'] = blocks.loc[plus_index, 'end']
        blocks.loc[minus_index, 'site'] = blocks.loc[minus_index, 'start']

    blocks.loc[plus_index,'start'] = blocks.loc[plus_index, 'site'] - upstream_distance
    blocks.loc[plus_index,'end'] = blocks.loc[plus_index, 'site'] + downstream_distance
    blocks.loc[minus_index,'end'] = blocks.loc[minus_index, 'site'] + upstream_distance
    blocks.loc[minus_index,'start'] = blocks.loc[minus_index, 'site'] - downstream_distance
    blocks = blocks[blocks['start'] >= 0]
    return blocks

    
def get_tss_region(gff, upstream_distance,downstream_distance):
    data = get_region_around_site(gff, [TRANSCRIPT_TYPE], upstream_distance,downstream_distance)
    return data


def get_cleavage_site_region(gff, upstream_distance, downstream_distance):
    data = get_region_around_site(gff, [TRANSCRIPT_TYPE], upstream_distance,downstream_distance, 
                                  is_start_location=False)
    return data


def get_donor_site_region(gff, upstream_distance, downstream_distance):
    data = get_region_around_site(gff, [INTRON_TYPE], upstream_distance,downstream_distance)
    return data


def get_acceptor_site_region(gff, upstream_distance, downstream_distance):
    data = get_region_around_site(gff, [INTRON_TYPE], upstream_distance,downstream_distance,
                                  is_start_location=False)
    return data


