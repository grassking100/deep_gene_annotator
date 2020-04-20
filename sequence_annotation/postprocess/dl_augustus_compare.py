import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.postprocess.utils import get_dl_folder_names, get_augustus_folder_names
from sequence_annotation.utils.utils import read_json, create_folder, batch_join
from sequence_annotation.utils.stats import exact_wilcox_rank_sum_compare as stats_test
from sequence_annotation.visual.boxplot import plot_boxplots

def plot_base_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    metrics = ['F1_exon','F1_intron','F1_other','macro_F1']
    metric_names = ['exon F1','intron F1','other F1','macro F1']
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,metrics,metric_names)
    
def plot_block_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    metrics = ['exon_F1','internal_exon_F1','intron_F1']+['intron_chain_F1','gene_F1']
    metric_names = ['exon F1','internal exon F1','intron F1']+['intron chain F1','gene F1']
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,metrics,metric_names)
    
def plot_site_matched_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    metrics = ['F1_TSS','F1_CA','F1_donor_site','F1_acceptor_site']
    metric_names = ['TSS F1','Cleavage site F1',
                    'Donor site F1','Acceptor site F1']
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,metrics,metric_names)
    
def plot_mean_abs_dist_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    metrics = ['mean_TSS','mean_CA','mean_donor_site',
               'mean_acceptor_site']
    metric_names = ['Mean abs. distance\nof TSS',
                    'Mean abs. distance\nof cleavage site',
                    'Mean abs. distance\nof donor site',
                    'Mean abs. distance\nof acceptor site']
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,metrics,metric_names)


def read_base_block_result(paths):
    result = {}
    for name, path in paths.items():
        result[name] = read_json(path)
    result = pd.DataFrame.from_dict(result).T
    return result


def read_base_result(paths):
    result = {}
    for name, path in paths.items():
        result[name] = read_json(path)
    result = pd.DataFrame.from_dict(result).T
    return result


def read_site_result(paths):
    result = {}
    for name, path in paths.items():
        result[name] = {}
        for method, data in read_json(path).items():
            for key, value in data.items():
                result[name]["{}_{}".format(method, key)] = value
    result = pd.DataFrame.from_dict(result).T
    return result


def compare_df(lhs_df, rhs_df, lhs_name, rhs_name, filter_names=None):
    result = {lhs_name: {}, rhs_name: {}}
    for name in lhs_df.columns:
        if filter_names is None or name in filter_names:
            l_values = list(lhs_df[name])
            r_values = list(rhs_df[name])
            result[lhs_name]["mean_{}".format(name)] = np.mean(l_values)
            result[rhs_name]["mean_{}".format(name)] = np.mean(r_values)
            result[lhs_name]["std_{}".format(name)] = np.std(l_values)
            result[rhs_name]["std_{}".format(name)] = np.std(r_values)
            result[lhs_name]["median_{}".format(name)] = np.std(l_values)
            result[rhs_name]["median_{}".format(name)] = np.std(r_values)
    result = pd.DataFrame.from_dict(result)
    return result


def write_tsv(df,root,name,index=False):
    path = os.path.join(root, name)
    df.to_csv(path, sep='\t', index=index)


class Comparer:
    def __init__(self,lhs_root,rhs_root,lhs_names,rhs_names,
                 lhs_rel_path,rhs_rel_path,lhs_source,rhs_source,
                 output_root):
        self.lhs_root = lhs_root
        self.rhs_root = rhs_root
        self.lhs_names = lhs_names
        self.rhs_names = rhs_names
        self.lhs_rel_path = lhs_rel_path
        self.rhs_rel_path = rhs_rel_path
        self.lhs_source = lhs_source
        self.rhs_source = rhs_source
        self.output_root = output_root

    def _read_base_block(self,name):
        lhs_paths = batch_join(self.lhs_root, self.lhs_names,
                               self.lhs_rel_path + '/'+name)
        rhs_paths = batch_join(self.rhs_root, self.rhs_names,
                               self.rhs_rel_path + '/'+name)
        lhs_result = read_base_block_result(lhs_paths)
        rhs_result = read_base_block_result(rhs_paths)
        return lhs_result,rhs_result
    
    def _read_site_result(self,name):
        lhs_paths = batch_join(self.lhs_root, self.lhs_names,
                               self.lhs_rel_path + '/'+name)
        rhs_paths = batch_join(self.rhs_root, self.rhs_names,
                               self.rhs_rel_path + '/'+name)
        lhs_result = read_site_result(lhs_paths)
        rhs_result = read_site_result(rhs_paths)
        return lhs_result,rhs_result

    def _compare(self,name,read_method,plot_method):
        lhs_result,rhs_result = read_method('{}.json'.format(name))
        stats = stats_test(lhs_result,rhs_result,self.lhs_source,
                           self.rhs_source,threshold=0.1)
        write_tsv(stats, self.output_root, '{}_stats.tsv'.format(name))
        plot_method(lhs_result,rhs_result,self.lhs_source,self.rhs_source)
        plt.savefig(os.path.join(self.output_root, '{}_boxplot.png').format(name))
        comp = compare_df(lhs_result,rhs_result,self.lhs_source,self.rhs_source)
        write_tsv(comp, self.output_root, '{}_comp.tsv'.format(name),index=True)
    
    def compare_base(self):
        self._compare('base_performance',self._read_base_block,
                      plot_base_boxplots)
        
    def compare_block(self):
        self._compare('block_performance',self._read_base_block,
                      plot_block_boxplots)
        
    def compare_p_a_abs_diff(self):
        self._compare('p_a_abs_diff',self._read_site_result,
                      plot_mean_abs_dist_boxplots)
        
    def compare_a_p_abs_diff(self):
        self._compare('a_p_abs_diff',self._read_site_result,
                      plot_mean_abs_dist_boxplots)
        
    def compare_site_matched(self):
        self._compare('site_matched',self._read_site_result,
                      plot_site_matched_boxplots)
        
    def compare(self):
        self.compare_base()
        self.compare_block()
        self.compare_p_a_abs_diff()
        self.compare_a_p_abs_diff()
        self.compare_site_matched()

    
def main(split_table_path, dl_root, aug_project_root,
         aug_folder_prefix, output_root):
    revised_to_raw = os.path.join(output_root,'revised_to_raw')
    raw_to_augustus = os.path.join(output_root,'raw_to_augustus')
    revised_to_augustus = os.path.join(output_root,'revised_to_augustus')
    create_folder(output_root)
    create_folder(revised_to_raw)
    create_folder(raw_to_augustus)
    create_folder(revised_to_augustus)
    split_table = pd.read_csv(split_table_path)
    dl_folder_names = get_dl_folder_names(split_table)
    aug_folder_names = get_augustus_folder_names(aug_folder_prefix, len(split_table))
    aug_rel_path='test/evaluate'
    dl_rel_path='testing/test'
    raw_dl_root = os.path.join(dl_root,'predicted')
    revised_dl_root = os.path.join(dl_root,'revised_test')

    #Compare raw DL and revised DL
    comparer = Comparer(raw_dl_root,revised_dl_root,
                        dl_folder_names,dl_folder_names,
                        dl_rel_path,dl_rel_path,
                        'origin DL','revised DL',revised_to_raw)
    comparer.compare()
    #Compare Augustus and raw DL
    comparer = Comparer(aug_project_root,raw_dl_root,
                        aug_folder_names,dl_folder_names,
                        aug_rel_path,dl_rel_path,
                        'Augustus','origin DL',raw_to_augustus)
    comparer.compare()
    #Compare Augustus and revised DL
    comparer = Comparer(aug_project_root,revised_dl_root,
                        aug_folder_names,dl_folder_names,
                        aug_rel_path,dl_rel_path,
                        'Augustus','revised DL',revised_to_augustus)
    comparer.compare()

if __name__ == '__main__':
    parser = ArgumentParser(description="Compare deep learnging result and augustus "
                            "result on testing data")
    parser.add_argument("-t", "--split_table_path", required=True,
                        help="The path of splitting information")
    parser.add_argument("-d", "--dl_root", required=True,
                        help="The root of deep learning revised result")
    parser.add_argument("-a", "--aug_project_root", required=True,
                        help="The root of saved Augustus project")
    parser.add_argument("-p", "--aug_folder_prefix", required=True,
                        help="The name or prefix of Augustus result folders")
    parser.add_argument("-s", "--output_root", required=True,
                        help="The root to save result file")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
