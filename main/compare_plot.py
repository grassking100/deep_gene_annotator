import os,sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/..")
from sequence_annotation.postprocess.utils import get_summary
from sequence_annotation.utils.utils import create_folder

def read(path):
    dl = pd.read_csv(path,index_col=0).T.reset_index(drop=True)
    return dl
def get_boxplot(data,columns,names):
    melted = data[columns+['Source']]
    melted.columns = names + ['Source']
    melted = melted.melt(id_vars='Source')
    plt.clf()
    sns.set(font_scale = 1.6)
    ax = sns.boxplot(x='variable', y='value', hue='Source', data=melted)
    ax.set(xlabel=None, ylabel=None)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, borderaxespad=0.)
    
def save_boxplots(data,root):
    ###
    base_columns = ['base_F1_exon', 'base_F1_intron', 'base_F1_other']
    macro_columns = ['base_macro_F1']
    block_columns = ['block_intron_F1','block_exon_F1']
    chained_block_columns = ['block_intron_chain_F1','block_gene_F1']
    site_boundary_columns = ['site_TSS','site_cleavage_site']
    site_splicing_columns = ['site_splicing_donor_site','site_splicing_acceptor_site']
    distance_boundary_columns = ['distance_TSS','distance_cleavage_site']
    distance_splicing_columns = ['distance_splicing_donor_site','distance_splicing_acceptor_site']
    base_names = ['Exon','Intron','Intergenic region']
    macro_names = ['Macro F-score']
    block_names = ['Intron block','Exon block']
    chained_block_names = ['Chained introns','Gene']
    site_boundary_names = ['TSS','CS']
    site_splicing_names = ['DS','AS']
    distance_boundary_names = ['TSS','CS']
    distance_splicing_names = ['DS','AS']
    ###
    get_boxplot(data,base_columns,base_names)
    plt.savefig(os.path.join(root,'base.png'), bbox_inches = 'tight',pad_inches = 0.1)
    
    get_boxplot(data,macro_columns,macro_names)
    plt.savefig(os.path.join(root,'macro.png'), bbox_inches = 'tight',pad_inches = 0.1)
    
    get_boxplot(data,site_boundary_columns,site_boundary_names)
    plt.savefig(os.path.join(root,'site_boundary.png'), bbox_inches = 'tight',pad_inches = 0.1)
    
    get_boxplot(data,site_splicing_columns,site_splicing_names)
    plt.savefig(os.path.join(root,'site_splicing.png'), bbox_inches = 'tight',pad_inches = 0.1)
    
    get_boxplot(data,distance_boundary_columns,distance_boundary_names)
    plt.savefig(os.path.join(root,'distance_boundary.png'), bbox_inches = 'tight',pad_inches = 0.1)
    
    get_boxplot(data,distance_splicing_columns,distance_splicing_names)
    plt.savefig(os.path.join(root,'distance_splicing.png'), bbox_inches = 'tight',pad_inches = 0.1)
    
    get_boxplot(data,block_columns,block_names)
    plt.savefig(os.path.join(root,'block.png'), bbox_inches = 'tight',pad_inches = 0.1)
    
    get_boxplot(data,chained_block_columns,chained_block_names)
    plt.savefig(os.path.join(root,'chained_block.png'), bbox_inches = 'tight',pad_inches = 0.1)

if __name__ == '__main__':    
    parser = ArgumentParser(description="Compare deep learnging result and augustus "
                            "result on testing data")
    parser.add_argument("-d", "--dl_root", required=True, help="The root of deep learning result")
    parser.add_argument("-a", "--augustus_root", required=True,help="The root of saved Augustus project")
    parser.add_argument("-r", "--revised_dl_root", required=True, help="The root of deep learning revised result")
    parser.add_argument("-o", "--output_root", required=True,help="The root to save result file")
    args = parser.parse_args()

    ###
    dl_test_summary_path = os.path.join(args.output_root,'dl_test_summary.csv')
    augustus_test_summary_path = os.path.join(args.output_root,'augustus_test_summary.csv')
    revised_dl_test_summary_path = os.path.join(args.output_root,'revised_dl_test_summary.csv')
    dl_to_augustus_path = os.path.join(args.output_root,'dl_to_augustus')
    revised_to_augustus_path = os.path.join(args.output_root,'revised_to_augustus')
    revised_to_raw_dl_path = os.path.join(args.output_root,'revised_to_raw_dl')
    ###
    create_folder(args.output_root)
    create_folder(dl_to_augustus_path)
    create_folder(revised_to_augustus_path)
    create_folder(revised_to_raw_dl_path)
    #
    get_summary(args.dl_root,'testing').to_csv(dl_test_summary_path)
    get_summary(args.augustus_root,'test').to_csv(augustus_test_summary_path)
    get_summary(args.revised_dl_root).to_csv(revised_dl_test_summary_path)
    #
    augustus = read(augustus_test_summary_path)
    augustus['Source']='Augustus'
    dl = read(dl_test_summary_path)
    dl['Source']='DL'
    revised_dl = read(revised_dl_test_summary_path)
    revised_dl['Source']='DL with revision'
    #
    data = pd.concat([augustus,dl],sort=True)
    save_boxplots(data,dl_to_augustus_path)

    data = pd.concat([augustus,revised_dl],sort=True)
    save_boxplots(data,revised_to_augustus_path)

    data = pd.concat([dl,revised_dl],sort=True)
    save_boxplots(data,revised_to_raw_dl_path)
