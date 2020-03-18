import numpy as np
from scipy import stats

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
    site_diff_['mode'] = {}
    types = ['TSS','CA','donor_site','acceptor_site']
    arrs = [transcript_start_diff_,transcript_end_diff_,
            donor_site_diff_,acceptor_site_diff_]
    for type_,arr in zip(types,arrs):
        if len(arr)>0:
            median_ = np.median(arr)
            mean_ = np.mean(arr)
            mode = stats.mode(arr)[0][0]
            if round_value is not None:
                mean_ = round(mean_,round_value)
            site_diff_['median'][type_] = float(median_)
            site_diff_['mean'][type_] = float(mean_)
            site_diff_['mode'][type_] = float(mode)
    return site_diff_
