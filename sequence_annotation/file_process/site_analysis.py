import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from multiprocessing import Pool
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import create_folder, write_json
from sequence_annotation.utils.stats import get_stats
from sequence_annotation.file_process.utils import read_gff, get_gff_with_intron,get_gff_with_attribute
from sequence_annotation.file_process.utils import INTRON_TYPE,TRANSCRIPT_TYPE

def _get_site_diff(ref_sites,compare_sites,absolute):
    ref_sites = np.array(list(ref_sites),dtype=float)
    compare_sites = np.array(list(compare_sites),dtype=float)
    dist = ref_sites.reshape(-1,1).repeat(len(compare_sites.T),axis=1) - compare_sites.reshape(1,-1)
    if absolute:
        diff = np.nanmin(np.abs(dist),axis=1)
    else:
        diff = dist[np.arange(len(dist)),np.nanargmin(np.abs(dist),axis=1)]
    return diff.tolist()

def get_site_diff(answer,predict,types=None,
                  five_end=True,absolute=True,
                  answer_as_ref=True,
                  multiprocess=None):
    if types is not None:
        answer = answer[answer['feature'].isin(types)]
        predict = predict[predict['feature'].isin(types)]

    chroms = set(list(answer['chr'])).intersection(set(list(predict['chr'])))
    answer = answer[['chr','strand','start','end']].copy()
    predict = predict[['chr','strand','start','end']].copy()
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
            if strand == '+':
                site_type = 'start' if five_end else 'end'
            else:
                site_type = 'end' if five_end else 'start'
            answer_sites = set(list(answer_group[site_type]))
            predict_sites = set(list(predict_group[site_type]))
            if answer_as_ref:
                ref_sites = answer_sites
                compare_sites = predict_sites
            else:
                ref_sites = predict_sites
                compare_sites = answer_sites
            kwarg_list.append((ref_sites,compare_sites,absolute))
    
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
    return get_site_diff(answer, predict, types=[INTRON_TYPE], **kwargs)


def get_acceptor_site_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, types=[INTRON_TYPE], five_end=False, **kwargs)


def get_transcript_start_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, types=[TRANSCRIPT_TYPE], **kwargs)


def get_transcript_end_diff(answer, predict, **kwargs):
    return get_site_diff(answer, predict, types=[TRANSCRIPT_TYPE], five_end=False, **kwargs)

def _get_site_id(data,site_type):
    return set(data['chr']+"_"+data['strand']+"_"+data[site_type].astype(str))

def get_site_ratio(answer,predict,types,five_end=True,get_recall=True):
    answer = answer[answer['feature'].isin(types)]
    predict = predict[predict['feature'].isin(types)]
    
    plus_answer = answer[answer['strand']=='+']
    minus_answer = answer[answer['strand']=='-']
    
    plus_predict = predict[predict['strand']=='+']
    minus_predict = predict[predict['strand']=='-']
    
    if five_end:
        answer_sites = _get_site_id(plus_answer,'start')
        answer_sites = answer_sites.union(_get_site_id(minus_answer,'end'))
        predict_sites = _get_site_id(plus_predict,'start')
        predict_sites = predict_sites.union(_get_site_id(minus_predict,'end'))
    else:
        answer_sites = _get_site_id(plus_answer,'end')
        answer_sites = answer_sites.union(_get_site_id(minus_answer,'start'))
        predict_sites = _get_site_id(plus_predict,'end')
        predict_sites = predict_sites.union(_get_site_id(minus_predict,'start'))

    result = float('nan')
    same_num = len(answer_sites.intersection(predict_sites))
    if get_recall:
        answer_num = len(answer_sites)
        if answer_num > 0:
            result = same_num/answer_num
    else:
        predict_num = len(predict_sites)
        if predict_num > 0:
            result = same_num/predict_num
    return result


def get_donor_site_ratio(answer, predict, **kwargs):
    return get_site_ratio(answer, predict,[INTRON_TYPE], **kwargs)


def get_acceptor_site_ratio(answer, predict, **kwargs):
    return get_site_ratio(answer,predict, [INTRON_TYPE],five_end=False,**kwargs)


def get_transcript_start_ratio(answer,predict,**kwargs):
    return get_site_ratio(answer, predict, [TRANSCRIPT_TYPE], **kwargs)


def get_transcript_end_ratio(answer, predict, **kwargs):
    return get_site_ratio(answer,predict, [TRANSCRIPT_TYPE],five_end=False,**kwargs)


def get_all_site_ratio(answer, predict, round_value=None):
    ratio = {'precision': {}, 'recall': {}, 'F1': {}}
    ratio['precision']['TSS'] = get_transcript_start_ratio(answer, predict, get_recall=False)
    ratio['precision']['cleavage_site'] = get_transcript_end_ratio(answer, predict, get_recall=False)
    ratio['precision']['splicing_donor_site'] = get_donor_site_ratio(answer, predict, get_recall=False)
    ratio['precision']['splicing_acceptor_site'] = get_acceptor_site_ratio(answer, predict, get_recall=False)

    ratio['recall']['TSS'] = get_transcript_start_ratio(answer, predict)
    ratio['recall']['cleavage_site'] = get_transcript_end_ratio(answer, predict)
    ratio['recall']['splicing_donor_site'] = get_donor_site_ratio(answer, predict)
    ratio['recall']['splicing_acceptor_site'] = get_acceptor_site_ratio(answer, predict)

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
    types = ['TSS', 'cleavage_site', 'splicing_donor_site', 'splicing_acceptor_site']
    arrs = [start_diff, end_diff, donor_diff, acceptor_diff]
    for type_, arr in zip(types, arrs):
        if return_value:
            site_diffs[type_] = arr
        else:
            if len(arr) > 0:
                result = get_stats(arr,round_value=round_value)
                for key,value in result.items():
                    if  key not in site_diffs:
                        site_diffs[key] = {}
                    site_diffs[key][type_] = value
    return site_diffs

def plot_site_diff(predict,answer,saved_root,answer_as_ref=True):
    create_folder(saved_root)
    predict = predict[predict['feature']!=INTRON_TYPE]
    answer = answer[answer['feature']!=INTRON_TYPE]
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


def main(predict_path, answer_path, output_root):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    plot_site_diff(predict,answer,output_root)
    plot_site_diff(predict,answer,output_root,answer_as_ref=False)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Show distance between predict GFF to answer GFF')
    parser.add_argument("-p","--predict_path",required=True,
                        help='The path of prediction result in GFF format')
    parser.add_argument("-a","--answer_path",required=True,
                        help='The path of answer result in GFF format')
    parser.add_argument("-o","--output_root",required=True,
                        help="Path to save result")
    args = parser.parse_args()

    config_path = os.path.join(args.saved_root, 'performance_setting.json')
    config = vars(args)

    main(**config)
