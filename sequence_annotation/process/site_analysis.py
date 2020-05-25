import os
import sys
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from multiprocessing import Pool
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder, read_gff
from sequence_annotation.utils.utils import get_gff_with_attribute,write_json
from sequence_annotation.preprocess.utils import get_gff_with_intron,INTRON_TYPES

def _get_site_diff(ref_sites,compare_sites,include_zero,absolute):
    ref_sites = np.array(ref_sites,dtype=float)
    compare_sites = np.array(compare_sites,dtype=float)
    dist = ref_sites.reshape(-1,1).repeat(len(compare_sites.T),axis=1) - compare_sites.reshape(1,-1)
    if not include_zero:
        dist[np.where(dist == 0)] = np.nan
    if absolute:
        diff = np.nanmin(np.abs(dist),axis=1)
    else:
        diff = dist[np.arange(len(dist)),np.nanargmin(np.abs(dist),axis=1)]
    return diff.tolist()

def get_site_diff(answer,predict,types,
                  is_start=True,absolute=True,
                  answer_as_ref=True,include_zero=True,
                  multiprocess=None):
    site_type = 'start' if is_start else 'end'
    answer = answer[answer['feature'].isin(types)].copy()
    predict = predict[predict['feature'].isin(types)].copy()
    chroms = set(list(answer['chr'])).intersection(set(list(predict['chr'])))
    answer['coord'] = answer['chr'] + "_" + answer['strand']
    predict['coord'] = predict['chr'] + "_" + predict['strand']
    answer = answer.groupby('coord')
    predict = predict.groupby('coord')
    kwarg_list = []
    for chrom in chroms:
        for strand in ['+', '-']:
            coord = chrom + "_" + strand
            if coord not in answer.groups or coord not in predict.groups:
                continue
            answer_group = answer.get_group(coord)
            predict_group = predict.get_group(coord)
            answer_sites = list(answer_group[site_type])
            predict_sites = list(predict_group[site_type])
            if answer_as_ref:
                ref_sites = answer_sites
                compare_sites = predict_sites
            else:
                ref_sites = predict_sites
                compare_sites = answer_sites
            kwarg_list.append((ref_sites,compare_sites,include_zero,absolute))
    
    if multiprocess is None:
        results = [_get_site_diff(*kwargs) for kwargs in kwarg_list]
    else:
        with Pool(processes=multiprocess) as pool:
            results = pool.starmap(_get_site_diff, kwarg_list)
        
    site_diffs = []
    for item in results:
        site_diffs += item

    return site_diffs


def get_donor_site_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, ['intron'], **kwargs)


def get_acceptor_site_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, ['intron'], is_start=False, **kwargs)


def get_transcript_start_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, ['mRNA'], **kwargs)


def get_transcript_end_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, ['mRNA'], is_start=False, **kwargs)


def get_site_matched_ratio(answer,predict,types,
                           is_start=True,answer_denominator=True):
    site_type = 'start' if is_start else 'end'
    answer = answer[answer['feature'].isin(types)]
    predict = predict[predict['feature'].isin(types)]
    answer_sites = set(answer['chr']+"_"+answer['strand']+"_"+answer[site_type].astype(str))
    predict_sites = set(predict['chr']+"_"+predict['strand']+"_"+predict[site_type].astype(str))
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


def get_acceptor_site_matched_ratio(answer, predict, **kwargs):
    return get_site_matched_ratio(answer,predict, ['intron'],is_start=False,
                                  **kwargs)


def get_transcript_start_matched_ratio(answer,predict,**kwargs):
    return get_site_matched_ratio(answer, predict, ['mRNA'], **kwargs)


def get_transcript_end_matched_ratio(answer, predict, **kwargs):
    return get_site_matched_ratio(answer,predict, ['mRNA'],is_start=False,
                                  **kwargs)


def get_all_site_matched_ratio(answer, predict, round_value=None):
    ratio = {'precision': {}, 'recall': {}, 'F1': {}}
    ratio['precision']['TSS'] = get_transcript_start_matched_ratio(
        answer, predict, answer_denominator=False)
    ratio['precision']['cleavage_site'] = get_transcript_end_matched_ratio(
        answer, predict, answer_denominator=False)
    ratio['precision']['splicing_donor_site'] = get_donor_site_matched_ratio(
        answer, predict, answer_denominator=False)
    ratio['precision']['splicing_acceptor_site'] = get_acceptor_site_matched_ratio(
        answer, predict, answer_denominator=False)

    ratio['recall']['TSS'] = get_transcript_start_matched_ratio(
        answer, predict)
    ratio['recall']['cleavage_site'] = get_transcript_end_matched_ratio(answer, predict)
    ratio['recall']['splicing_donor_site'] = get_donor_site_matched_ratio(
        answer, predict)
    ratio['recall']['splicing_acceptor_site'] = get_acceptor_site_matched_ratio(
        answer, predict)

    for type_ in ['TSS', 'cleavage_site', 'splicing_donor_site', 'splicing_acceptor_site']:
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
    types = ['TSS', 'cleavage_site', 'splicing_donor_site', 'splicing_acceptor_site']
    arrs = [start_diff, end_diff, donor_diff, acceptor_diff]
    for type_, arr in zip(types, arrs):
        if return_value:
            site_diffs[type_] = arr
        else:
            if len(arr) > 0:
                median_ = np.nanmedian(arr)
                mean_ = np.nanmean(arr)
                mode = stats.mode(arr,nan_policy='omit')[0][0]
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
        site_txt = ' '.join(type_.split('_'))
        if answer_as_ref:
            plt.title("The distance between closest predicted {} to answer {}".format(site_txt, site_txt))
        else:
            plt.title("The distance between closest answer {} to predicted {}".format(site_txt, site_txt))
        plt.ylabel("Number")
        plt.xlabel("Distance")
        plt.hist(diffs, bins=100, log=True)
        if answer_as_ref:
            plt.savefig(os.path.join(saved_root, "{}_p_a_diff_distribution.png".format(type_)))
            write_json(diffs,os.path.join(saved_root, "{}_p_a_diff.json".format(type_)))
        else:
            plt.savefig(os.path.join(saved_root, "{}_a_p_diff_distribution.png".format(type_)))
            write_json(diffs,os.path.join(saved_root, "{}_a_p_diff.json".format(type_)))
    
def main(predict_path, answer_path, saved_root):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    plot_site_diff(predict,answer,saved_root)
    plot_site_diff(predict,answer,saved_root,answer_as_ref=False)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Show distance between predict GFF to answer GFF')
    parser.add_argument("-p","--predict_path",required=True,
                        help='The path of prediction result in GFF format')
    parser.add_argument("-a","--answer_path",required=True,
                        help='The path of answer result in GFF format')
    parser.add_argument("-s","--saved_root",required=True,
                        help="Path to save result")
    args = parser.parse_args()

    config_path = os.path.join(args.saved_root, 'performance_setting.json')
    config = vars(args)

    main(**config)
