import os
import pandas as pd
from ..utils.utils import get_file_name, read_json


def get_dl_folder_names(split_table):
    dl_folder_names = []
    for item in split_table.to_dict('record'):
        folder_name = "{}_{}".format(get_file_name(item['training_path']),
                                     get_file_name(item['validation_path']))
        dl_folder_names.append(folder_name)
    return dl_folder_names


def get_augustus_folder_names(aug_folder_prefix, num):
    aug_folder_names = []
    for index in range(num):
        folder_name = "{}_{}".format(aug_folder_prefix, index + 1)
        aug_folder_names.append(folder_name)
    return aug_folder_names


def _get_min_threshold(params, alpha=None):
    threshold = None
    alpha = alpha or 4
    weights = params['weights']
    means = params['means']
    stds = params['stds']
    for index in range(len(weights)):
        mean = means[index]
        std = stds[index]
        new_threshold = mean - std * alpha
        if threshold is None:
            threshold = new_threshold
        else:
            threshold = min(new_threshold, threshold)
    return threshold


def get_min_threshold(params, alpha=None):
    threshold = pow(10, _get_min_threshold(params, alpha))
    return threshold

def get_summary(root,data_type=None):
    result = {}
    types = ['base','block','distance','site']
    file_names = ['base_performance.json','block_performance.json','abs_diff.json','site_matched.json']
    columns = [
        'base_F1_exon','base_F1_intron','base_F1_other','base_macro_F1',
        'block_exon_F1','block_gene_F1','block_intron_F1','block_intron_chain_F1',
        'distance_TSS','distance_cleavage_site','distance_splicing_acceptor_site','distance_splicing_donor_site',
        'site_TSS','site_cleavage_site','site_splicing_acceptor_site','site_splicing_donor_site'
    ]
    for type_,file_name in zip(types,file_names):
        for name in sorted(os.listdir(root)):
            if data_type is not None:
                path = os.path.join(root,name,data_type,file_name)
            else:
                path = os.path.join(root,name,file_name)
            if os.path.exists(path):
                data_ = read_json(path)
                data = {}
                if type_ == 'site':
                    for key,value in data_['F1'].items():
                        data[type_+"_"+key] = value
                elif type_ == 'distance':
                    for key,value in data_['mean'].items():
                        data[type_+"_"+key] = value    
                else:
                    for key,value in data_.items():
                        data[type_+"_"+key] = value

                if name not in result:
                    result[name] = {}
                result[name].update(data)
    result = pd.DataFrame.from_dict(result).T[columns].T
    return result