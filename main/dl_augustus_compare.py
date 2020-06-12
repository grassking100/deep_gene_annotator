import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__) + "/..")
from sequence_annotation.postprocess.utils import get_dl_folder_names, get_augustus_folder_names
from sequence_annotation.utils.utils import read_json, create_folder, batch_join,write_json
from sequence_annotation.utils.stats import exact_wilcox_rank_sum_compare as stats_test
from sequence_annotation.visual.boxplot import plot_boxplots

BASE_METRICS = ['F1_exon','F1_intron','F1_other','macro_F1']
BASE_NAMES = ['Exon F1','Intron F1','Intergeneic region F1','Macro F1']
BLOCK_METRICS = ['exon_F1','intron_F1']
BLOCK_NAMES = ['Exon block F1','Intron block F1']
BLOCK_CHAIN_METRICS = ['intron_chain_F1','gene_F1']
BLOCK_CHAIN_NAMES = ['Chained introns F1','Gene F1']
SITE_MATCHED_METRICS = ['F1_TSS','F1_cleavage_site','F1_splicing_donor_site','F1_splicing_acceptor_site']
SITE_MATCHED_NAMES = ['TSS F1','Cleavage site F1','Donor site F1','Acceptor site F1']
ABS_DIST_METRICS = ['mean_TSS','mean_cleavage_site','mean_splicing_donor_site','mean_splicing_acceptor_site']
ABS_DIST_NAMES = ['Mean distance of TSS',
                  'Mean distance of cleavage site',
                  'Mean distance of splicing donor site',
                  'Mean distance of splicing acceptor site']

def plot_base_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,BASE_NAMES,'F1')
    
def plot_block_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,BLOCK_NAMES+BLOCK_CHAIN_NAMES,'F1')
    
def plot_site_matched_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,SITE_MATCHED_NAMES,'F1')
    
def plot_mean_abs_dist_boxplots(lhs_result,rhs_result,lhs_name,rhs_name):
    plot_boxplots(lhs_result,rhs_result,lhs_name,
                  rhs_name,ABS_DIST_NAMES,'distance')


def read_base_block_result(paths):
    result = {}
    for name, path in paths.items():
        result[name] = read_json(path)
    result = pd.DataFrame.from_dict(result).T
    result = result.rename(columns=dict(zip(BASE_METRICS,BASE_NAMES)))
    result = result.rename(columns=dict(zip(BLOCK_METRICS,BLOCK_NAMES)))
    result = result.rename(columns=dict(zip(BLOCK_CHAIN_METRICS,BLOCK_CHAIN_NAMES)))
    return result


def read_site_result(paths):
    result = {}
    for name, path in paths.items():
        result[name] = {}
        for method, data in read_json(path).items():
            for key, value in data.items():
                result[name]["{}_{}".format(method, key)] = value
    result = pd.DataFrame.from_dict(result).T
    result = result.rename(columns=dict(zip(SITE_MATCHED_METRICS,SITE_MATCHED_NAMES)))
    result = result.rename(columns=dict(zip(ABS_DIST_METRICS,ABS_DIST_NAMES)))
    return result


def compare_df(lhs_df, rhs_df, lhs_name, rhs_name, filter_names=None):
    result = {lhs_name: {}, rhs_name: {}}
    for name in lhs_df.columns:
        if filter_names is None or name in filter_names:
            l_values = list(lhs_df[name])
            r_values = list(rhs_df[name])
            result[lhs_name]["mean of {}".format(name)] = np.mean(l_values)
            result[rhs_name]["mean of {}".format(name)] = np.mean(r_values)
            result[lhs_name]["std of {}".format(name)] = np.std(l_values)
            result[rhs_name]["std of {}".format(name)] = np.std(r_values)
            result[lhs_name]["median of {}".format(name)] = np.median(l_values)
            result[rhs_name]["median of {}".format(name)] = np.median(r_values)
    result = pd.DataFrame.from_dict(result)
    return result


def write_tsv(df,root,name,index=False):
    path = os.path.join(root, name)
    df.to_csv(path, sep='\t', index=index)


class Comparer:
    def __init__(self,lhs_root,rhs_root,lhs_names,rhs_names,
                 lhs_rel_path,rhs_rel_path,
                 lhs_source,rhs_source,output_root):
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
        lhs_name = rhs_name = name
        if self.lhs_rel_path is not None:
            lhs_name = os.path.join(self.lhs_rel_path ,name)
        if self.rhs_rel_path is not None:
            rhs_name = os.path.join(self.rhs_rel_path ,name)
        lhs_paths = batch_join(self.lhs_root, self.lhs_names,
                               lhs_name)
        rhs_paths = batch_join(self.rhs_root, self.rhs_names,
                               rhs_name)
        lhs_result = read_base_block_result(lhs_paths)
        rhs_result = read_base_block_result(rhs_paths)
        return lhs_result,rhs_result
    
    def _read_site_result(self,name):
        lhs_name = rhs_name = name
        if self.lhs_rel_path is not None:
            lhs_name = os.path.join(self.lhs_rel_path ,name)
        if self.rhs_rel_path is not None:
            rhs_name = os.path.join(self.rhs_rel_path ,name)
        lhs_paths = batch_join(self.lhs_root, self.lhs_names,
                               lhs_name)
        rhs_paths = batch_join(self.rhs_root, self.rhs_names,
                               rhs_name)
        lhs_result = read_site_result(lhs_paths)
        rhs_result = read_site_result(rhs_paths)
        return lhs_result,rhs_result

    def _compare(self,name,lhs_result,rhs_result,plot_method):
        stats = stats_test(lhs_result,rhs_result,self.lhs_source,
                           self.rhs_source,threshold=0.05)
        write_tsv(stats, self.output_root, '{}_stats.tsv'.format(name))
        plot_method(lhs_result,rhs_result,self.lhs_source,self.rhs_source)
        plt.savefig(os.path.join(self.output_root, '{}_boxplot.png').format(name))
        comp = compare_df(lhs_result,rhs_result,self.lhs_source,self.rhs_source)
        write_tsv(comp, self.output_root, '{}_comp.tsv'.format(name),index=True)
        return stats
    
    def compare_base(self):
        lhs_result,rhs_result = self._read_base_block('base_performance.json')
        return self._compare('base_performance',lhs_result,rhs_result,
                             plot_base_boxplots)
        
    def compare_block(self):
        lhs_result,rhs_result = self._read_base_block('block_performance.json')
        return self._compare('block_performance',lhs_result,rhs_result,
                             plot_block_boxplots)
        
    def compare_p_a_abs_diff(self):
        lhs_result,rhs_result = self._read_site_result('p_a_abs_diff.json')
        return self._compare('p_a_abs_diff',lhs_result,rhs_result,
                             plot_mean_abs_dist_boxplots)
        
    def compare_a_p_abs_diff(self):
        lhs_result,rhs_result = self._read_site_result('a_p_abs_diff.json')
        return self._compare('a_p_abs_diff',lhs_result,rhs_result,
                             plot_mean_abs_dist_boxplots)
    
    def compare_abs_diff(self):
        lhs_result,rhs_result = self._read_site_result('abs_diff.json')
        return self._compare('abs_diff',lhs_result,rhs_result,
                             plot_mean_abs_dist_boxplots)
        
    def compare_site_matched(self):
        lhs_result,rhs_result = self._read_site_result('site_matched.json')
        return self._compare('site_matched',lhs_result,rhs_result,
                             plot_site_matched_boxplots)
        
    def compare(self):
        base_df = self.compare_base()
        block_df = self.compare_block()
        dist_df = self.compare_abs_diff()
        self.compare_p_a_abs_diff()
        self.compare_a_p_abs_diff()
        site_matched_df = self.compare_site_matched()
        base_df['type'] = 'Base'
        block_df.loc[block_df['metric'].isin(BLOCK_NAMES),'type'] = 'Block'
        block_df.loc[block_df['metric'].isin(BLOCK_CHAIN_NAMES),'type'] = 'Chained blocks'
        dist_df['type'] = 'Distance'
        site_matched_df['type'] = 'Site'
        base_targets = base_df[base_df['metric'].isin(BASE_NAMES)]
        block_targets = block_df[block_df['metric'].isin(BLOCK_NAMES+BLOCK_CHAIN_NAMES)]
        dist_targets = dist_df[dist_df['metric'].isin(ABS_DIST_NAMES)]
        site_matched_targets = site_matched_df[site_matched_df['metric'].isin(SITE_MATCHED_NAMES)]
        targets = pd.concat([base_targets,block_targets,dist_targets,site_matched_targets])
        pivot = pd.pivot_table(targets, values='pass', index=['type'],
                               columns=['compare'], aggfunc=np.sum,
                               fill_value=0)
        pivot.reset_index(level=0, inplace=True)
        write_tsv(targets,self.output_root,'compare.tsv')
        write_tsv(pivot,self.output_root,'pivot.tsv')
        win_num = sum(pivot[pivot['type']=='Distance']['less'])
        win_num += sum(pivot[pivot['type']!='Distance']['greater'])
        loss_num = sum(pivot[pivot['type']=='Distance']['greater'])
        loss_num += sum(pivot[pivot['type']!='Distance']['less'])
        equal_num = len(targets) - win_num - loss_num
        write_json({'win':win_num,'loss':loss_num,'equal':equal_num},
                   os.path.join(self.output_root,'compare_num.json'))
        
    
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
    aug_rel_path='test'
    raw_dl_rel_path='testing'
    dl_rel_path=None
    raw_dl_root = os.path.join(dl_root,'predicted')
    revised_dl_root = os.path.join(dl_root,'revised_test')

    #Compare raw DL and revised DL
    comparer = Comparer(raw_dl_root,revised_dl_root,
                        dl_folder_names,dl_folder_names,
                        raw_dl_rel_path,dl_rel_path,
                        'origin DL','revised DL',revised_to_raw)
    comparer.compare()
    #Compare Augustus and raw DL
    comparer = Comparer(aug_project_root,raw_dl_root,
                        aug_folder_names,dl_folder_names,
                        aug_rel_path,raw_dl_rel_path,
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
