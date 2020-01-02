import os
import numpy as np
import pandas as pd
from .utils import read_json

def site_diff(answer,predict,types,is_start=True,absolute=True,answer_as_ref=True):
    site_type = 'start' if is_start else 'end'
    answer = answer[answer['feature'].isin(types)]
    predict = predict[predict['feature'].isin(types)]
    chroms = set(list(answer['chr'])).intersection(set(list(predict['chr'])))
    site_diff = []
    for chrom in chroms:
        for strand in ['+','-']:
            answer_group = answer[(answer['chr']==chrom) & (answer['strand']==strand)]
            predict_group = predict[(predict['chr']==chrom) & (predict['strand']==strand)]
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
                    if absolute:
                        diff = min([abs(compare_site-ref_site) for compare_site in compare_sites])
                    else:
                        if strand == '-':
                            diff = min([ref_site-compare_site for compare_site in compare_sites])
                        else:
                            diff = min([compare_site-ref_site for compare_site in compare_sites])
                    site_diff.append(diff)
    return site_diff
    
def donor_site_diff(answer,predict,**kwargs):
    return site_diff(answer,predict,['intron'],**kwargs)

def acceptor_site_diff(answer,predict,absolute=True,**kwargs):
    return site_diff(answer,predict,['intron'],is_start=False,**kwargs)

def transcript_start_diff(answer,predict,absolute=True,**kwargs):
    return site_diff(answer,predict,['mRNA'],**kwargs)

def transcript_end_diff(answer,predict,absolute=True,**kwargs):
    return site_diff(answer,predict,['mRNA'],is_start=False,**kwargs)
    
def site_abs_diff(answer,predict,round_value=None,**kwargs):
    transcript_start_diff_ = transcript_start_diff(answer,predict,**kwargs)
    transcript_end_diff_ = transcript_end_diff(answer,predict,**kwargs)
    donor_site_diff_ = donor_site_diff(answer,predict,**kwargs)
    acceptor_site_diff_ = acceptor_site_diff(answer,predict,**kwargs)
    site_diff_ = {}
    site_diff_['median'] = {}
    site_diff_['mean'] = {}
    types = ['TSS','CA','donor_site','acceptor_site']
    arrs = [transcript_start_diff_,transcript_end_diff_,
            donor_site_diff_,acceptor_site_diff_]
    for type_,arr in zip(types,arrs):
        median_ = np.median(arr)
        mean_ = np.mean(arr)
        if round_value is not None:
            mean_ = round(mean_,round_value)
        site_diff_['median'][type_] = median_
        site_diff_['mean'][type_] = mean_
    return site_diff_
    
def site_diff_table(roots,names):
    site_diff = []
    error_paths = []
    for name,root in zip(names,roots):
        try:
            p_a_abs_diff = read_json(os.path.join(root,'p_a_abs_diff.json'))
            a_p_abs_diff = read_json(os.path.join(root,'a_p_abs_diff.json'))
        except:
            error_paths.append(root)
            continue
        for key,values in p_a_abs_diff.items():
            for target,value in values.items():
                site_diff_ = {}
                site_diff_['method'] = '{}(abs(predict-answer))'.format(key)
                site_diff_['target'] = target
                site_diff_['value'] = value
                site_diff_['name'] = name
                site_diff.append(site_diff_)

        for key,values in a_p_abs_diff.items():
            for target,value in values.items():
                site_diff_ = {}
                site_diff_['method'] = '{}(abs(answer-predict))'.format(key)
                site_diff_['target'] = target
                site_diff_['value'] = value
                site_diff_['name'] = name
                site_diff.append(site_diff_)
    site_diff = pd.DataFrame.from_dict(site_diff)[['name','target','method','value']]
    return site_diff,error_paths

def block_performance_table(roots,names):
    block_performance = []
    error_paths = []
    for name,root in zip(names,roots):
        try:
            data = read_json(os.path.join(root,'block_performance.json'))
        except:
            error_paths.append(root)
            continue
            
        for target,value in data.items():
            block_performance_ = {}
            block_performance_['target'] = target
            block_performance_['value'] = value
            block_performance_['name'] = name
            block_performance.append(block_performance_)

    block_performance = pd.DataFrame.from_dict(block_performance)
    columns = list(block_performance.columns)
    columns.remove('name')
    columns = ['name'] + columns
    return block_performance[columns],error_paths

def get_length(gff,type_):
    block = gff[gff['feature']==type_]
    lengths = block['end']-block['start']+1
    return list(lengths)
