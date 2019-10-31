import os
import numpy as np
import pandas as pd
from .utils import read_gff,read_gffcompare_stats

def site_diff(answer,predict,types,is_start=True,absolute=True):
    if (answer['strand'] != '+').any() or (predict['strand'] != '+').any():
        raise Exception("Wrong strand")
    site_type = 'start' if is_start else 'end'
    answer = answer[answer['feature'].isin(types)]
    chroms = set(list(answer['chr']))
    answer = answer.groupby('chr')
    predict = predict[predict['feature'].isin(types)].groupby('chr')
    site_diff = []
    for chrom in chroms:
        answer_group = answer.get_group(chrom)
        try:
            predict_group = predict.get_group(chrom)
            answer_sites = list(answer_group[site_type])
            predict_sites = list(predict_group[site_type])
            for predict_site in predict_sites:
                if absolute:
                    diff = min([abs(predict_site-site) for site in answer_sites])
                else:
                    diff = None
                    for site in answer_sites:
                        diff_ = predict_site-site
                        if diff is None or abs(diff) > abs(diff_):
                            diff = diff_
                site_diff.append(diff)
        except KeyError:
            pass
    return site_diff
    
def donor_site_diff(answer,predict,absolute=True):
    return site_diff(answer,predict,['intron'],absolute=absolute)
def accept_site_diff(answer,predict,absolute=True):
    return site_diff(answer,predict,['intron'],is_start=False,absolute=absolute)
def transcript_start_diff(answer,predict,absolute=True):
    return site_diff(answer,predict,['mRNA'],absolute=absolute)
def transcript_end_diff(answer,predict,absolute=True):
    return site_diff(answer,predict,['mRNA'],is_start=False,absolute=absolute)
    
def site_diff_table(roots,names):
    site_diff_ = {}
    for name,root in zip(names,roots):
        #if os.path.exists(root):
        answer_path = os.path.join(root,'answers.gff3')
        predict_path = os.path.join(root,'test_gffcompare_1.gff3')
        try:
            answer = read_gff(answer_path)
            predict = read_gff(predict_path)
            transcript_start_diff_ = transcript_start_diff(answer,predict)
            transcript_end_diff_ = transcript_end_diff(answer,predict)
            donor_site_diff_ = donor_site_diff(answer,predict)
            accept_site_diff_ = accept_site_diff(answer,predict)
            site_diff_[name] = {}
            site_diff_[name]['TSS'] = np.median(transcript_start_diff_)
            site_diff_[name]['CA'] = np.median(transcript_end_diff_)
            site_diff_[name]['donor_site'] = np.median(donor_site_diff_)
            site_diff_[name]['accept_site'] = np.median(accept_site_diff_)
        except:
            print(name)
    return pd.DataFrame.from_dict(site_diff_)

def miss_nove_sensitivity_precision_table(roots,names):
    sensitivity = {}
    precision = {}
    miss_novel_stats = {}
    
    for name,root in zip(names,roots):
        path = os.path.join(root,'test_gffcompare_1.stats')
        print(path)
        if os.path.exists(path):
            sensitivity_,precision_,miss_novel_stats_ = read_gffcompare_stats(path)
            for level in sensitivity_:
                if level not in sensitivity:
                    sensitivity[level] = {}
                if level not in precision:
                    precision[level] = {}
                sensitivity[level][name] = sensitivity_[level]
                precision[level][name] = precision_[level]   
            for level in miss_novel_stats_:
                if level not in miss_novel_stats:
                    miss_novel_stats[level] = {}
                miss_novel_stats[level][name] = miss_novel_stats_[level]
    miss_novel_stats = pd.DataFrame.from_dict(miss_novel_stats)
    precision = pd.DataFrame.from_dict(precision)
    sensitivity = pd.DataFrame.from_dict(sensitivity)
    return miss_novel_stats,sensitivity,precision