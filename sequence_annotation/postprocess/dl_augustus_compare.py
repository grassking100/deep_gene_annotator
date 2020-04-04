import os,sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_json, create_folder, batch_join
from sequence_annotation.postprocess.utils import get_dl_folder_names, get_augustus_folder_names

def read_base_block_results(paths):
    results = {}
    for name,path in paths.items():
        results[name] = read_json(path)
    results = pd.DataFrame.from_dict(results)
    return results

def read_base_results(paths):
    results = {}
    for name,path in paths.items():
        results[name] = read_json(path)
    results = pd.DataFrame.from_dict(results)
    return results

def read_site_results(paths):
    results = {}
    for name,path in paths.items():
        results[name] = {}
        for method,result in read_json(path).items():
            for key,value in result.items():
                results[name]["{}_{}".format(method,key)] = value
    results = pd.DataFrame.from_dict(results)
    return results
    
def compare_df(lhs_df,rhs_df,lhs_name,rhs_name,filter_names=None,show_std=True):
    t_lhs_df = lhs_df.T
    t_rhs_df = rhs_df.T
    compared_result = {lhs_name:{},rhs_name:{}}
    for name in t_lhs_df.columns:
        if filter_names is None or any([filter_name in name for filter_name in filter_names]):
            l_values = list(t_lhs_df[name])
            r_values = list(t_rhs_df[name])
            compared_result[lhs_name]["mean_{}".format(name)] = np.mean(l_values)
            compared_result[rhs_name]["mean_{}".format(name)] = np.mean(r_values)
            if show_std:
                compared_result[lhs_name]["std_{}".format(name)] = np.std(l_values)
                compared_result[rhs_name]["std_{}".format(name)] = np.std(r_values)
    compared_result = pd.DataFrame.from_dict(compared_result)
    if not show_std:
        diff = compared_result[lhs_name]-compared_result[rhs_name]
        compared_result = compared_result.assign(diff=diff)
    return compared_result

def main(split_table_path,dl_revised_root,aug_project_root,
         aug_folder_prefix,output_root):
    
    create_folder(output_root)
    dl_rel_path = 'test'
    #dl_rel_path = 'testing/test'
    split_table = pd.read_csv(split_table_path)
    dl_folder_names = get_dl_folder_names(split_table)
    aug_folder_names = get_augustus_folder_names(aug_folder_prefix,len(split_table))

    #Create paths of deep learning result
    dl_block_performance_paths = batch_join(dl_revised_root,dl_folder_names,
                                            dl_rel_path+'/block_performance.json')
    dl_base_performance_paths = batch_join(dl_revised_root,dl_folder_names,
                                           dl_rel_path+'/base_performance.json')
    dl_p_a_abs_diff_paths = batch_join(dl_revised_root,dl_folder_names,
                                       dl_rel_path+'/p_a_abs_diff.json')
    dl_a_p_abs_diff_paths = batch_join(dl_revised_root,dl_folder_names,
                                       dl_rel_path+'/a_p_abs_diff.json')
    
    dl_site_matched_paths = batch_join(dl_revised_root,dl_folder_names,
                                       dl_rel_path+'/site_matched.json')
    
    #Create paths of Augustus result
    aug_block_performance_paths = batch_join(aug_project_root,aug_folder_names,
                                             'test/evaluate/block_performance.json')
    aug_base_performance_paths = batch_join(aug_project_root,aug_folder_names,
                                            'test/evaluate/base_performance.json')
    aug_p_a_abs_diff_paths = batch_join(aug_project_root,aug_folder_names,
                                        'test/evaluate/p_a_abs_diff.json')
    aug_a_p_abs_diff_paths = batch_join(aug_project_root,aug_folder_names,
                                        'test/evaluate/a_p_abs_diff.json')
    aug_site_matched_paths = batch_join(aug_project_root,aug_folder_names,
                                        'test/evaluate/site_matched.json')
    
    #Read deep learning result
    dl_block_results = read_base_block_results(dl_block_performance_paths)
    dl_base_results = read_base_block_results(dl_base_performance_paths)
    dl_p_a_abs_diff_results = read_site_results(dl_p_a_abs_diff_paths)
    dl_a_p_abs_diff_results = read_site_results(dl_a_p_abs_diff_paths)
    dl_site_matched_results = read_site_results(dl_site_matched_paths)
    #Read Augustus result
    aug_block_results = read_base_block_results(aug_block_performance_paths)
    aug_base_results = read_base_block_results(aug_base_performance_paths)
    aug_p_a_abs_diff_results = read_site_results(aug_p_a_abs_diff_paths)
    aug_a_p_abs_diff_results = read_site_results(aug_a_p_abs_diff_paths)
    aug_site_matched_results = read_site_results(aug_site_matched_paths)
    #Compare result between Augustus and deep learning
    p_a_abs_diff_compare_result = compare_df(dl_p_a_abs_diff_results,aug_p_a_abs_diff_results,'DL','Augustus',show_std=False)
    a_p_abs_diff_compare_result = compare_df(dl_a_p_abs_diff_results,aug_a_p_abs_diff_results,'DL','Augustus',show_std=False)
    base_compare_result = compare_df(dl_base_results,aug_base_results,'DL','Augustus',show_std=False)
    block_compare_result = compare_df(dl_block_results,aug_block_results,'DL','Augustus',show_std=False)
    site_matched_result = compare_df(dl_site_matched_results,aug_site_matched_results,'DL','Augustus',show_std=False)
    #Save result
    path = os.path.join(output_root,'a_p_abs_diff_compare_result.tsv')
    a_p_abs_diff_compare_result.to_csv(path,sep='\t')
    
    path = os.path.join(output_root,'p_a_abs_diff_compare_result.tsv')
    p_a_abs_diff_compare_result.to_csv(path,sep='\t')
    
    path = os.path.join(output_root,'base_compare_result.tsv')
    base_compare_result.to_csv(path,sep='\t')
    
    path = os.path.join(output_root,'block_compare_result.tsv')
    block_compare_result.to_csv(path,sep='\t')
    
    path = os.path.join(output_root,'site_matched_result.tsv')
    site_matched_result.to_csv(path,sep='\t')
    
if __name__ =='__main__':
    parser = ArgumentParser(description='Compare deep learnging result and augustus result on testing data')
    parser.add_argument("-t","--split_table_path",help="The path of splitting information",required=True)
    parser.add_argument("-d","--dl_revised_root",help="The root of deep learning revised result",required=True)
    parser.add_argument("-a","--aug_project_root",help="The root of saved Augustus project",required=True)
    parser.add_argument("-p","--aug_folder_prefix",help="The name or prefix of Augustus result folders",required=True)
    parser.add_argument("-s","--output_root",help="The root to save result file",required=True)
    args = parser.parse_args()    
    kwargs = vars(args)    
    main(**kwargs)
