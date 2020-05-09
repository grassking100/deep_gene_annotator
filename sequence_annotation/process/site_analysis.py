import os
import sys
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder, read_gff
from sequence_annotation.utils.utils import get_gff_with_attribute
from sequence_annotation.preprocess.utils import get_gff_with_intron,INTRON_TYPES

def get_site_diff(answer,predict,types,
                  is_start=True,absolute=True,
                  answer_as_ref=True,include_zero=True):
    site_type = 'start' if is_start else 'end'
    answer = answer[answer['feature'].isin(types)]
    predict = predict[predict['feature'].isin(types)]
    chroms = set(list(answer['chr'])).intersection(set(list(predict['chr'])))
    site_diffs = []
    for chrom in chroms:
        for strand in ['+', '-']:
            answer_group = answer[(answer['chr'] == chrom)
                                  & (answer['strand'] == strand)]
            predict_group = predict[(predict['chr'] == chrom)
                                    & (predict['strand'] == strand)]
            answer_sites = list(answer_group[site_type])
            predict_sites = list(predict_group[site_type])
            if len(answer_sites) > 0 and len(predict_sites) > 0:
                if answer_as_ref:
                    ref_sites = answer_sites
                    compare_sites = predict_sites
                else:
                    ref_sites = predict_sites
                    compare_sites = answer_sites
                for ref_site in ref_sites:
                    diffs = [
                        compare_site - ref_site
                        for compare_site in compare_sites
                    ]
                    
                    if not include_zero:
                        diffs = [diff for diff in diffs if diff != 0]
                    if len(diffs) > 0:
                        if absolute:
                            diff = min([abs(diff) for diff in diffs])
                        else:
                            diffs = np.array(diffs)
                            if strand == '-':
                                diffs = -diffs
                            index = abs(diffs).argmin()
                            diff = diffs[index]
                        site_diffs.append(diff)
    return site_diffs


def get_donor_site_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, ['intron'], **kwargs)


def get_acceptor_site_diff(answer, predict, absolute=True, **kwargs):
    return get_site_diff(answer, predict, ['intron'], is_start=False, **kwargs)


def get_transcript_start_diff(answer, predict, absolute=True, **kwargs):
    return get_site_diff(answer, predict, ['mRNA'], **kwargs)


def get_transcript_end_diff(answer, predict, absolute=True, **kwargs):
    return get_site_diff(answer, predict, ['mRNA'], is_start=False, **kwargs)


def get_site_matched_ratio(answer,predict,types,
                           is_start=True,answer_denominator=True):
    site_type = 'start' if is_start else 'end'
    answer = answer[answer['feature'].isin(types)]
    predict = predict[predict['feature'].isin(types)]
    chroms = set(list(answer['chr'])).intersection(set(list(predict['chr'])))
    answer_sites = []
    predict_sites = []
    for chrom in chroms:
        for strand in ['+', '-']:
            answer_group = answer[(answer['chr'] == chrom)
                                  & (answer['strand'] == strand)]
            predict_group = predict[(predict['chr'] == chrom)
                                    & (predict['strand'] == strand)]
            answer_sites += [
                "{}_{}_{}".format(chrom, strand, site)
                for site in list(answer_group[site_type])
            ]
            predict_sites += [
                "{}_{}_{}".format(chrom, strand, site)
                for site in list(predict_group[site_type])
            ]

    answer_sites = set(answer_sites)
    predict_sites = set(predict_sites)
    intersection_sites = answer_sites.intersection(predict_sites)
    ratio = 0
    if answer_denominator:
        if len(answer_sites) > 0:
            ratio = len(intersection_sites) / len(answer_sites)
        else:
            ratio = float('nan')
    elif len(predict_sites) > 0:
        ratio = len(intersection_sites) / len(predict_sites)
    return ratio


def get_donor_site_matched_ratio(answer, predict, **kwargs):
    return get_site_matched_ratio(answer, predict, ['intron'], **kwargs)


def get_acceptor_site_matched_ratio(answer, predict, absolute=True, **kwargs):
    return get_site_matched_ratio(answer,predict, ['intron'],is_start=False,
                                  **kwargs)


def get_transcript_start_matched_ratio(answer,predict,absolute=True,**kwargs):
    return get_site_matched_ratio(answer, predict, ['mRNA'], **kwargs)


def get_transcript_end_matched_ratio(answer, predict, absolute=True, **kwargs):
    return get_site_matched_ratio(answer,predict, ['mRNA'],is_start=False,
                                  **kwargs)


def get_all_site_matched_ratio(answer, predict, round_value=None):
    ratio = {'precision': {}, 'recall': {}, 'F1': {}}
    ratio['precision']['TSS'] = get_transcript_start_matched_ratio(
        answer, predict, answer_denominator=False)
    ratio['precision']['CA'] = get_transcript_end_matched_ratio(
        answer, predict, answer_denominator=False)
    ratio['precision']['donor_site'] = get_donor_site_matched_ratio(
        answer, predict, answer_denominator=False)
    ratio['precision']['acceptor_site'] = get_acceptor_site_matched_ratio(
        answer, predict, answer_denominator=False)

    ratio['recall']['TSS'] = get_transcript_start_matched_ratio(
        answer, predict)
    ratio['recall']['CA'] = get_transcript_end_matched_ratio(answer, predict)
    ratio['recall']['donor_site'] = get_donor_site_matched_ratio(
        answer, predict)
    ratio['recall']['acceptor_site'] = get_acceptor_site_matched_ratio(
        answer, predict)

    for type_ in ['TSS', 'CA', 'donor_site', 'acceptor_site']:
        precision = ratio['precision'][type_]
        recall = ratio['recall'][type_]
        if precision + recall == 0:
            ratio['F1'][type_] = 0
        else:
            ratio['F1'][type_] = 2 * precision * recall / (precision + recall)

    for method, dict_ in ratio.items():
        for key, value in dict_.items():
            if round_value is not None:
                value = round(value, round_value)
            ratio[method][key] = float(value)
    return ratio


def get_all_site_diff(answer,predict,round_value=None,
                      return_value=False,**kwargs):
    start_diff = get_transcript_start_diff(answer, predict, **kwargs)
    end_diff = get_transcript_end_diff(answer, predict, **kwargs)
    donor_diff = get_donor_site_diff(answer, predict, **kwargs)
    acceptor_diff = get_acceptor_site_diff(answer, predict, **kwargs)
    site_diffs = {}
    if not return_value:
        site_diffs['median'] = {}
        site_diffs['mean'] = {}
        site_diffs['mode'] = {}
    types = ['TSS', 'CA', 'donor_site', 'acceptor_site']
    arrs = [start_diff, end_diff, donor_diff, acceptor_diff]
    for type_, arr in zip(types, arrs):
        if return_value:
            site_diffs[type_] = list(arr)
        else:
            if len(arr) > 0:
                median_ = np.median(arr)
                mean_ = np.mean(arr)
                mode = stats.mode(arr)[0][0]
                if round_value is not None:
                    mean_ = round(mean_, round_value)
                site_diffs['median'][type_] = float(median_)
                site_diffs['mean'][type_] = float(mean_)
                site_diffs['mode'][type_] = float(mode)
    return site_diffs


def plot_site_diff(predict,answer,saved_root,answer_as_ref=True):
    create_folder(saved_root)
    predict = predict[~predict['feature'].isin(INTRON_TYPES)]
    answer = answer[~answer['feature'].isin(INTRON_TYPES)]
    predict = get_gff_with_intron(get_gff_with_attribute(predict))
    answer = get_gff_with_intron(get_gff_with_attribute(answer))
    site_diffs = get_all_site_diff(answer,predict,return_value=True,
                                   absolute=False,answer_as_ref=answer_as_ref)
    for type_, diffs in site_diffs.items():
        plt.clf()
        if answer_as_ref:
            plt.title("The distance between nearest predict {} to real {}".format(type_, type_))
        else:
            plt.title("The distance between nearest real {} to predict {}".format(type_, type_))
        plt.ylabel("Number")
        plt.xlabel("Distance")
        plt.hist(diffs, bins=100, log=True)
        if answer_as_ref:
            plt.savefig(os.path.join(saved_root, "{}_p_a_diff_distribution.png".format(type_)))
        else:
            plt.savefig(os.path.join(saved_root, "{}_a_p_diff_distribution.png".format(type_)))
    
def main(predict_path, answer_path, saved_root):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    plot_site_diff(predict,answer,saved_root)
    plot_site_diff(predict,answer,saved_root,answer_as_ref=False)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Show distance between predict GFF to answer GFF')
    parser.add_argument("-p","--predict_path",
                        help='The path of prediction result in GFF format',
                        required=True)
    parser.add_argument("-a","--answer_path",
                        help='The path of answer result in GFF format',
                        required=True)
    parser.add_argument("-s","--saved_root",
                        help="Path to save result",
                        required=True)
    args = parser.parse_args()

    config_path = os.path.join(args.saved_root, 'performance_setting.json')
    config = vars(args)

    main(**config)
