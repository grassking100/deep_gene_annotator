import os, sys
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.file_process.utils import TRANSCRIPT_TYPE, INTRON_TYPE
from sequence_annotation.file_process.utils import get_gff_with_intron,get_gff_with_updated_attribute

def get_region_around_site(gff, upstream_distance, downstream_distance, is_start_location=True):
    if len(set(gff['strand']) - set(['+', '-'])) > 0:
        raise Exception("Invalid strand")
    gff = gff[['chr', 'strand', 'start', 'end','parent']].copy().sort_values(['chr', 'start', 'end','strand'])
    gff['transcript_source'] = gff['parent']
    gff['site'] = None
    gff['id'] = ["site_{}".format(i) for i in range(1,1+len(gff))]
    gff['score'] = '.'
    plus_index = gff[(gff['strand'] == '+')].index
    minus_index = gff[(gff['strand'] == '-')].index
    if is_start_location:
        gff.loc[plus_index, 'site'] = gff.loc[plus_index, 'start']
        gff.loc[minus_index, 'site'] = gff.loc[minus_index, 'end']
    else:
        gff.loc[plus_index, 'site'] = gff.loc[plus_index, 'end']
        gff.loc[minus_index, 'site'] = gff.loc[minus_index, 'start']

    gff.loc[plus_index,'start'] = gff.loc[plus_index, 'site'] - upstream_distance
    gff.loc[plus_index,'end'] = gff.loc[plus_index, 'site'] + downstream_distance
    gff.loc[minus_index,'end'] = gff.loc[minus_index, 'site'] + upstream_distance
    gff.loc[minus_index,'start'] = gff.loc[minus_index, 'site'] - downstream_distance
    gff = gff[gff['start'] >= 0]
    gff = get_gff_with_updated_attribute(gff)
    return gff

    
def get_tss_region(gff, upstream_distance,downstream_distance):
    gff = gff[gff['feature']==TRANSCRIPT_TYPE]
    data = get_region_around_site(gff,upstream_distance,downstream_distance)
    return data


def get_cleavage_site_region(gff, upstream_distance, downstream_distance):
    gff = gff[gff['feature']==TRANSCRIPT_TYPE]
    data = get_region_around_site(gff, upstream_distance,downstream_distance, is_start_location=False)
    return data


def get_donor_site_region(gff, upstream_distance, downstream_distance):
    if INTRON_TYPE not in set(gff['feature']):
        gff = get_gff_with_intron(gff)
    gff = gff[gff['feature']==INTRON_TYPE]
    data = get_region_around_site(gff,upstream_distance,downstream_distance)
    return data


def get_acceptor_site_region(gff, upstream_distance, downstream_distance):
    if INTRON_TYPE not in set(gff['feature']):
        gff = get_gff_with_intron(gff)
    gff = gff[gff['feature']==INTRON_TYPE]
    data = get_region_around_site(gff, upstream_distance,downstream_distance, is_start_location=False)
    return data


