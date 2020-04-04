import os,sys
import pandas as pd
import numpy as np
from sklearn import mixture
from scipy import stats
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_gff, create_folder,get_gff_with_attribute
from sequence_annotation.preprocess.utils import GENE_TYPES,INTRON_TYPES,EXON_TYPES,get_gff_with_intron

def get_length(gff,features):
    gff = gff[gff['feature'].isin(features)]
    return np.array(list(gff['end']-gff['start']+1))

def norm_fit_log10(data,component_num=None):
    data = np.log10(data)
    component_num = component_num or 1
    if component_num==1:
        mean,std = stats.norm.fit(data)
        weights = [1]
        means = [mean]
        stds = [std]
    else:
        clf = mixture.GaussianMixture(n_components=component_num,
                                      covariance_type='spherical',
                                      max_iter=1000)
        clf.fit(data.reshape(-1,1))
        weights = clf.weights_
        means = clf.means_.flatten()
        stds = np.sqrt(clf.covariances_)
    params = {
        'weights':weights,
        'means':means,
        'stds':stds
    }
    return params

def get_norm_pdf(params,range_,merge=True):
    weights = params['weights']
    means = params['means']
    stds = params['stds']
    pdfs = []
    for index in range(len(weights)):
        weight = weights[index]
        mean = means[index]
        std = stds[index]
        pdf = stats.norm.pdf(range_,mean,std)*weight
        pdfs.append(pdf)
    pdfs = np.array(pdfs)
    if merge:
        pdf_sum = pdfs.sum(0)
        return pdf_sum
    else:       
        return pdfs

def get_range_of_log10(data,step=None):
    step = step or 1000
    log_data = np.log10(data)
    range_ = np.arange(min(log_data)*step,max(log_data)*step)/step
    return range_

def plot_log_hist(data,params,merge=False,title=None):
    plt.clf()
    range_ = get_range_of_log10(data)
    pdfs = get_norm_pdf(params,range_,merge)
    plt.plot(range_,pdfs.sum(0),label='gaussian model summation')
    plt.xlabel('log10(length)')
    plt.ylabel('densitiy')
    if title is not None:
        plt.title(title)
    for index,pdf in enumerate(pdfs):
        plt.plot(range_,pdf,label='gaussian model {}'.format(index+1))
    plt.hist(np.log10(data),bins=100,density=True)
    plt.legend()

def main(gff_path,output_root,component_num=None):
    create_folder(output_root)
    data_types = {'gene':GENE_TYPES,'intron':INTRON_TYPES,'exon':EXON_TYPES}
    gff = get_gff_with_attribute(read_gff(gff_path))
    gff = gff[gff['feature']!='intron']
    gff = get_gff_with_intron(gff)
    gaussian_models = []
    for type_,features in data_types.items():
        lengths = get_length(gff,features)
        params = norm_fit_log10(lengths,component_num)
        for weight,mean,std in zip(params['weights'],params['means'],params['stds']):
            gaussian_models.append({'weights':weight,'mean':mean,'std':std,'type':type_})

        title = 'The log10(length) distribution and gaussian model of {}'.format(type_)
        plot_log_hist(lengths,params,title=title)
        path = os.path.join(output_root,'{}_log10_length_gaussian_model'.format(type_))
        plt.savefig(path)

    gaussian_models = pd.DataFrame.from_dict(gaussian_models)[['type','weights','mean','std']]
    gaussian_model_path = os.path.join(output_root,'length_log10_gaussian_model.tsv')
    gaussian_models.to_csv(gaussian_model_path,sep='\t',index=None)
    
if __name__ =='__main__':
    parser = ArgumentParser(description="Create table about length statistic result about GFF data")
    parser.add_argument("-i","--gff_path",help="The path of input GFF to be analysized",required=True)
    parser.add_argument("-o","--output_root",help="The root to save result file",required=True)
    parser.add_argument("-n","--component_num",help="The number of Gaussian mixture model to fit",type=int,default=1)
    args = parser.parse_args()
    main(**vars(args))
