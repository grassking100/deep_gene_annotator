import os, sys
import pandas as pd
import numpy as np
from sklearn import mixture
from scipy import stats
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import read_gff, create_folder, get_gff_with_attribute
from sequence_annotation.preprocess.utils import GENE_TYPES, INTRON_TYPES, EXON_TYPES, get_gff_with_intron


def get_length(gff, features):
    gff = gff[gff['feature'].isin(features)]
    return np.array(list(gff['end'] - gff['start'] + 1))


def norm_fit_log10(data, component_num=None):
    data = np.log10(data)
    component_num = component_num or 1
    if component_num == 1:
        mean, std = stats.norm.fit(data)
        weights = [1]
        means = [mean]
        stds = [std]
    else:
        clf = mixture.GaussianMixture(n_components=component_num,
                                      covariance_type='spherical',
                                      max_iter=1000)
        clf.fit(data.reshape(-1, 1))
        weights = clf.weights_
        means = clf.means_.flatten()
        stds = np.sqrt(clf.covariances_)
    params = {'weights': weights, 'means': means, 'stds': stds}
    return params


def get_norm_pdf(params, range_, merge=True):
    weights = params['weights']
    means = params['means']
    stds = params['stds']
    pdfs = []
    for index in range(len(weights)):
        weight = weights[index]
        mean = means[index]
        std = stds[index]
        pdf = stats.norm.pdf(range_, mean, std) * weight
        pdfs.append(pdf)
    pdfs = np.array(pdfs)
    if merge:
        pdf_sum = pdfs.sum(0)
        return pdf_sum
    else:
        return pdfs


def get_range_of_log10(data, step=None):
    step = step or 1000
    log_data = np.log10(data)
    range_ = np.arange(min(log_data) * step, max(log_data) * step) / step
    return range_


def plot_log_hist(data, params=None, merge=False, title=None,density=True):
    plt.clf()
    range_ = get_range_of_log10(data)
    if params is not None:
        pdfs = get_norm_pdf(params, range_, merge)
        plt.plot(range_, pdfs.sum(0), label='model summation')
        for index, pdf in enumerate(pdfs):
            plt.plot(range_, pdf, label='model {}'.format(index + 1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.hist(np.log10(data), bins=100, density=density)
    plt.xlabel('log10(length)', fontsize=16)
    plt.ylabel('density', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=16)
    plt.legend(fontsize=12)


def main(gff_path, output_root, component_num=None):
    create_folder(output_root)
    data_types = {
        'gene': GENE_TYPES,
        'intron': INTRON_TYPES,
        'exon': EXON_TYPES
    }
    gaussian_model_path = os.path.join(output_root,'length_log10_gaussian_model.tsv')
    gff = get_gff_with_attribute(read_gff(gff_path))
    gff = gff[gff['feature'] != 'intron']
    gff = get_gff_with_intron(gff)
    
    type_lengths = {}
    for type_, features in data_types.items():
        lengths = get_length(gff, features)
        type_lengths[type_] = lengths
    
    if os.path.exists(gaussian_model_path):
        gaussian_models = pd.read_csv(gaussian_model_path, sep='\t')
        gaussian_model_groups = gaussian_models.groupby('types')
        gaussian_params = {}
        for type_ in list(gaussian_model_groups.groups):
            gaussian_params[type_] = gaussian_model_groups.get_group(type_).to_dict('list')
        
    else:
        gaussian_models = []
        gaussian_params = {}
        for type_ in data_types.keys():
            lengths = type_lengths[type_]
            params = norm_fit_log10(lengths, component_num)
            gaussian_params[type_] = params
            for weight, mean, std in zip(params['weights'], params['means'],
                                         params['stds']):
                gaussian_models.append({
                    'weights': weight,
                    'means': mean,
                    'stds': std,
                    'types': type_
                })

        gaussian_models = pd.DataFrame.from_dict(gaussian_models)[['types', 'weights', 'means', 'stds']]
        gaussian_models.to_csv(gaussian_model_path, sep='\t', index=None)
        
    for type_ in data_types.keys():
        lengths = type_lengths[type_]
        params = gaussian_params[type_]
        title = 'The log distribution and Gaussian model of {}\'s length (nt)'.format(type_)
        plot_log_hist(lengths, params, title=title)
        path = os.path.join(output_root,'{}_log10_length_gaussian_model'.format(type_))
        plt.savefig(path,bbox_inches = 'tight',pad_inches = 0)



if __name__ == '__main__':
    parser = ArgumentParser(
        description="Create table about length statistic result about GFF data"
    )
    parser.add_argument("-i","--gff_path",required=True,
                        help="The path of input GFF to be analysized")
    parser.add_argument("-o","--output_root",required=True,
                        help="The root to save result file")
    parser.add_argument("-n","--component_num",type=int,default=1,
                        help="The number of Gaussian mixture model to fit")
    args = parser.parse_args()
    main(**vars(args))
