import sys
import os
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import write_bed, read_gff, get_gff_with_attribute, create_folder
from sequence_annotation.preprocess.utils import get_gff_with_intron, INTRON_TYPES, RNA_TYPES


def get_region_around_site(gff, feature_types, upstream_distance,
                           downstream_distance, is_start_location=True):
    blocks = gff[gff['feature'].isin(feature_types)]
    if len(set(blocks['strand']) - set(['+', '-'])) > 0:
        raise Exception("Invalid strand")
    blocks = blocks[['chr', 'strand', 'start', 'end','parent']].copy()
    blocks['transcript_source'] = blocks['parent']
    blocks['site'] = None
    blocks['id'] = '.'
    blocks['score'] = '.'
    plus_index = blocks[(blocks['strand'] == '+')].index
    minus_index = blocks[(blocks['strand'] == '-')].index
    if is_start_location:
        blocks.loc[plus_index, 'site'] = blocks.loc[plus_index, 'start']
        blocks.loc[minus_index, 'site'] = blocks.loc[minus_index, 'end']
    else:
        blocks.loc[plus_index, 'site'] = blocks.loc[plus_index, 'end']
        blocks.loc[minus_index, 'site'] = blocks.loc[minus_index, 'start']

    blocks.loc[plus_index,
               'start'] = blocks.loc[plus_index, 'site'] - upstream_distance
    blocks.loc[plus_index,
               'end'] = blocks.loc[plus_index, 'site'] + downstream_distance
    blocks.loc[minus_index,
               'end'] = blocks.loc[minus_index, 'site'] + upstream_distance
    blocks.loc[minus_index,
               'start'] = blocks.loc[minus_index, 'site'] - downstream_distance
    blocks = blocks[blocks['start'] >= 0]
    blocks = blocks[~blocks[['chr', 'strand', 'start', 'end']].duplicated()]
    return blocks


def get_transcription_start_site_region(gff, upstream_distance,
                                        downstream_distance):
    data = get_region_around_site(gff, RNA_TYPES, upstream_distance,
                                  downstream_distance)
    return data


def get_cleavage_site_region(gff, upstream_distance, downstream_distance):
    data = get_region_around_site(gff, RNA_TYPES, upstream_distance,
                                  downstream_distance, 
                                  is_start_location=False)
    return data


def get_donor_site_region(gff, upstream_distance, downstream_distance):
    data = get_region_around_site(gff, INTRON_TYPES, upstream_distance,
                                  downstream_distance)
    return data


def get_acceptor_site_region(gff, upstream_distance, downstream_distance):
    data = get_region_around_site(gff, INTRON_TYPES, upstream_distance,
                                  downstream_distance,
                                  is_start_location=False)
    return data


def main(gff_path, output_root, tss_radius, cleavage_radius, donor_radius,
         acceptor_radius):
    create_folder(output_root)
    TSS_SIGNAL_RADIUS = 3
    CS_SIGNAL_RADIUS = 3
    gff = read_gff(gff_path)
    gff = gff[~gff['feature'].isin(INTRON_TYPES)]
    gff = get_gff_with_attribute(gff)
    gff = get_gff_with_intron(gff, update_attribute=False)
    tss_region = get_transcription_start_site_region(gff, tss_radius,
                                                     tss_radius)
    cs_region = get_cleavage_site_region(gff, cleavage_radius, cleavage_radius)
    donor_region = get_donor_site_region(gff, donor_radius, donor_radius)
    acceptor_region = get_acceptor_site_region(gff, acceptor_radius,
                                               acceptor_radius)
    tss_signal_region = get_transcription_start_site_region(
        gff, TSS_SIGNAL_RADIUS, TSS_SIGNAL_RADIUS)
    cs_signal_region = get_cleavage_site_region(gff, CS_SIGNAL_RADIUS,
                                                CS_SIGNAL_RADIUS)
    donor_signal_region = get_donor_site_region(gff, 0, 1)
    acceptor_signal_region = get_acceptor_site_region(gff, 1, 0)

    regions = [
        tss_region, cs_region, donor_region, acceptor_region,
        tss_signal_region, cs_signal_region, donor_signal_region,
        acceptor_signal_region
    ]

    names = [
        "tss_around_{}".format(tss_radius),
        "cleavage_site_around_{}".format(cleavage_radius),
        "donor_site_around_{}".format(donor_radius),
        "acceptor_site_around_{}".format(acceptor_radius), "tss_signal",
        "cleavage_site_signal", "donor_site_signal", "acceptor_site_signal"
    ]

    for region, name in zip(regions, names):
        write_bed(region, os.path.join(output_root, "{}.bed".format(name)))


if __name__ == '__main__':
    parser = ArgumentParser(description="This program will read gff file "
                            "and output block information in BED format")
    parser.add_argument("-i", "--gff_path",required=True,
                        help="Path of GFF file")
    parser.add_argument("-o", "--output_root", required=True,
                        help="Root to save data")
    parser.add_argument("-t", "--tss_radius", required=True, type=int)
    parser.add_argument("-c", "--cleavage_radius", required=True, type=int)
    parser.add_argument("-d", "--donor_radius", required=True, type=int)
    parser.add_argument("-a", "--acceptor_radius", required=True, type=int)

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
