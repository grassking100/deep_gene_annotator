import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, get_gff_with_attribute, get_gff_with_feature_coord, read_gff
from sequence_annotation.preprocess.utils import get_gff_with_intergenic_region,get_gff_with_intron,INTRON_TYPES,read_region_table
from sequence_annotation.preprocess.length_gaussian_modeling import norm_fit_log10,plot_log_hist

def get_feature_stats(gff,region_table,chrom_source):
    gff = gff[~gff['feature'].isin(INTRON_TYPES)]
    print("Get feature")
    gff = get_gff_with_attribute(gff)
    print("Get intron")
    gff = get_gff_with_intron(gff)
    print("Get intergenic region")
    gff = get_gff_with_intergenic_region(gff,region_table,chrom_source)
    print("Get feature and coordinate")
    gff = get_gff_with_feature_coord(gff)
    print("Start calculate")
    feature_group = gff.groupby('feature')
    stats = []
    length_dict = {}

    for feature,group in feature_group:
        lengths = list(group['end']-group['start']+1)
        stats_item = {'feature':feature}
        stats_item['number'] = len(group)
        stats_item['max'] = max(lengths)
        stats_item['min'] = min(lengths)
        stats_item['mean'] = np.mean(lengths)
        stats_item['std'] = np.std(lengths)
        stats_item['median'] = np.median(lengths)
        length_dict[feature] = lengths
        stats.append(stats_item)
    return stats,length_dict
        
def plot_length(length_dict,saved_root):
    for feature,lengths in length_dict.items():
        plt.figure()
        plt.hist(lengths,100)
        plt.title("The distribution of \n{}'s lengths (nt)".format(feature), fontsize=14)
        plt.xlabel("length", fontsize=14)
        plt.ylabel("number", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(saved_root,'{}_length.png'.format(feature).replace(' ','_')),
                   bbox_inches = 'tight',pad_inches = 0)
        
def plot_log_length(length_dict,saved_root,with_gaussian=False,predicted_mode=False,is_small_figure=False):
    component_nums = {'intron':2,'exon':2,'intergenic region':3,'gene':2,'mRNA':2}
    if predicted_mode:
        for key,value in component_nums.items():
            component_nums[key] += 1
        
    for feature,lengths in length_dict.items():
        params = None
        title = "The log distribution of \n{}'s lengths (nt)".format(feature)
        if with_gaussian:
            model_path = os.path.join(saved_root,"{}_models.tsv").format(feature.replace(' ','_'))
            title = 'The distribution and Gaussian model\n of {}\'s log-length (nt)'.format(feature)
            if False:#os.path.exists(model_path):
                params = pd.read_csv(model_path,sep='\t').to_dict('list')
            else:
                if feature in component_nums:
                    component_num = component_nums[feature]
                else:
                    component_num = 1
                params = norm_fit_log10(lengths, component_num)
                pd.DataFrame.from_dict(params).to_csv(model_path,sep='\t')
        if is_small_figure:
            plt.figure(figsize=(5,4))
            plot_log_hist(lengths, params,show_legend=False)
        else:
            plot_log_hist(lengths, params, title=title,show_legend=with_gaussian)
        plt.savefig(os.path.join(saved_root,'{}_log10_length.png'.format(feature).replace(' ','_')),
                    bbox_inches = 'tight',pad_inches = 0)
    
def main(gff_path,region_table_path,saved_root,chrom_source,**kwargs):
    create_folder(saved_root)
    gff = read_gff(gff_path)
    region_table = read_region_table(region_table_path)
    stats,lengths = get_feature_stats(gff,region_table,chrom_source)
    stats_df = pd.DataFrame.from_dict(stats)
    stats_df.to_csv(os.path.join(saved_root,'feature_stats.tsv'),sep='\t',index=None)
    plot_length(lengths,saved_root)
    plot_log_length(lengths,saved_root,**kwargs)
    
if __name__ =='__main__':
    parser = ArgumentParser(description="This program will write statistic data about GFF file")
    parser.add_argument("-i", "--gff_path", help="Path of input gff file",required=True)
    parser.add_argument("-r", "--region_table_path", help="Path of region table",required=True)
    parser.add_argument("-o", "--saved_root", help="Path of output file",required=True)
    parser.add_argument("-c","--chrom_source", type=str,required=True)
    parser.add_argument("-g","--with_gaussian", action='store_true')
    parser.add_argument("-p","--predicted_mode", action='store_true')
    parser.add_argument("-s","--is_small_figure", action='store_true')
    
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
