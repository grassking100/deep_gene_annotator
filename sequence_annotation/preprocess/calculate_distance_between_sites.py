import os, sys
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_bed, read_gff, InvalidStrandType


def _compare_site_distance(compared_sites, comparing_sites, compared_site_name,
                           comparing_site_name):
    strands = set(compared_sites['strand'])
    if strands != set(['+', '-']):
        raise InvalidStrandType(strands)
    data = []
    for strand in strands:
        print(strand)
        selected_compared_sites = set(compared_sites[
            compared_sites['strand'] == strand][compared_site_name])
        selected_comparing_sites = set(comparing_sites[
            comparing_sites['strand'] == strand][comparing_site_name])
        selected_compared_sites = np.array(
            sorted(list(selected_compared_sites)))
        selected_comparing_sites = np.array(
            sorted(list(selected_comparing_sites)))
        for site in selected_compared_sites:
            diff = selected_comparing_sites - site
            abs_diff = np.abs(diff)
            value = diff[abs_diff.argmin()]
            if strand == '-':
                value = -value
            data.append(value)
    return data


def compare_site_distance(exp_sites, ref_sites, exp_site_name, ref_site_name):
    r_to_e_diff = []
    e_to_r_diff = []
    exp_sites[exp_site_name] = exp_sites[exp_site_name].astype(float)
    ref_sites[ref_site_name] = ref_sites[ref_site_name].astype(float)
    for chr_ in set(exp_sites['chr']):
        print(chr_)
        selected_exp_sites = exp_sites[(exp_sites['chr'] == chr_)]
        selected_ref_sites = ref_sites[(ref_sites['chr'] == chr_)]
        r_to_e_diff += _compare_site_distance(exp_sites, ref_sites,
                                              exp_site_name, ref_site_name)
        e_to_r_diff += _compare_site_distance(ref_sites, exp_sites,
                                              ref_site_name, exp_site_name)
    return r_to_e_diff, e_to_r_diff


def calculate_stats(type_, site_diff):
    diff = {}
    diff['type'] = type_
    diff['min'] = min(site_diff)
    diff['max'] = max(site_diff)
    diff['median'] = np.median(site_diff)
    diff['mean'] = np.mean(site_diff)
    diff['std'] = np.std(site_diff)
    diff['mode'] = stats.mode(site_diff)[0][0]

    return diff


def plot_hist(title, path, values):
    plt.figure()
    plt.hist(values, 100)
    plt.title(title)
    plt.xlabel("distance")
    plt.ylabel("number")
    plt.savefig(path)


if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-b",
                        "--bed_path",
                        help="Path of selected official gene info file",
                        required=True)
    parser.add_argument("-t",
                        "--tss_path",
                        help="Path of TSS file",
                        required=True)
    parser.add_argument("-c",
                        "--cs_path",
                        help="Path of CS file",
                        required=True)
    parser.add_argument("-s",
                        "--saved_root",
                        help="Path to save",
                        required=True)
    args = parser.parse_args()

    ###Read file###
    reference_bed = read_bed(args.bed_path)
    tss_sites = read_gff(args.tss_path)
    cleavage_sites = read_gff(args.cs_path)
    ##
    tss_site_r_to_e_diff, tss_site_e_to_r_diff = compare_site_distance(
        tss_sites, reference_bed, 'start', 'five_end')
    cleavage_site_r_to_e_diff, cleavage_site_e_to_r_diff = compare_site_distance(
        cleavage_sites, reference_bed, 'start', 'three_end')
    stats_list = []
    stats_list.append(
        calculate_stats('tss_site_r_to_e_diff', tss_site_r_to_e_diff))
    stats_list.append(
        calculate_stats('tss_site_e_to_r_diff', tss_site_e_to_r_diff))
    stats_list.append(
        calculate_stats('cleavage_site_r_to_e_diff',
                        cleavage_site_r_to_e_diff))
    stats_list.append(
        calculate_stats('cleavage_site_e_to_r_diff',
                        cleavage_site_e_to_r_diff))
    pd.DataFrame.from_dict(stats_list).to_csv(os.path.join(
        args.saved_root, "site_diff.tsv"),
                                              index=False,
                                              sep='\t')

    plot_hist(
        'The distance from the closest reference TSS to experimental TSS',
        os.path.join(args.saved_root, 'tss_site_r_to_e_diff.png'),
        tss_site_r_to_e_diff)

    plot_hist(
        'The distance from the closest experimental TSS to reference TSS',
        os.path.join(args.saved_root, 'tss_site_e_to_r_diff.png'),
        tss_site_e_to_r_diff)

    plot_hist('The distance from the closest reference CS to experimental CS',
              os.path.join(args.saved_root, 'cleavage_site_r_to_e_diff.png'),
              cleavage_site_r_to_e_diff)

    plot_hist('The distance from the closest experimental CS to reference CS',
              os.path.join(args.saved_root, 'cleavage_site_e_to_r_diff.png'),
              cleavage_site_e_to_r_diff)
