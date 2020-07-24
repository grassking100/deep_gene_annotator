import os,sys
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
def read(path):
    dl = pd.read_csv(path,index_col=0).T.reset_index(drop=True)
    return dl
def get_boxplot(data,columns,names):
    melted = data[columns+['Source']]
    melted.columns = names + ['Source']
    melted = melted.melt(id_vars='Source')
    plt.clf()
    sns.set(font_scale = 1.3)
    ax = sns.boxplot(x='variable', y='value', hue='Source', data=melted)
    ax.set(xlabel=None, ylabel=None)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, borderaxespad=0.)
    
def save_boxplots(data,root):
    get_boxplot(data,base_columns,base_names)
    plt.savefig(os.path.join(root,'base.png'), bbox_inches = 'tight',
    pad_inches = 0.1)
    get_boxplot(data,macro_columns,macro_names)
    plt.savefig(os.path.join(root,'macro.png'), bbox_inches = 'tight',
    pad_inches = 0.1)
    get_boxplot(data,site_boundary_columns,site_boundary_names)
    plt.savefig(os.path.join(root,'site_boundary.png'), bbox_inches = 'tight',
    pad_inches = 0.1)
    get_boxplot(data,site_splicing_columns,site_splicing_names)
    plt.savefig(os.path.join(root,'site_splicing.png'), bbox_inches = 'tight',
    pad_inches = 0.1)
    get_boxplot(data,distance_boundary_columns,distance_boundary_names)
    plt.savefig(os.path.join(root,'distance_boundary.png'), bbox_inches = 'tight',
    pad_inches = 0.1)
    get_boxplot(data,distance_splicing_columns,distance_splicing_names)
    plt.savefig(os.path.join(root,'distance_splicing.png'), bbox_inches = 'tight',
    pad_inches = 0.1)
    
base_columns = ['base_F1_exon', 'base_F1_intron', 'base_F1_other']
macro_columns = ['base_macro_F1']
block_columns = ['block_exon_F1', 'block_intron_F1', 'block_gene_F1', 'block_intron_chain_F1']
site_boundary_columns = ['site_TSS','site_cleavage_site']
site_splicing_columns = ['site_splicing_donor_site','site_splicing_acceptor_site']
distance_boundary_columns = ['distance_TSS','distance_cleavage_site']
distance_splicing_columns = ['distance_splicing_donor_site','distance_splicing_acceptor_site']

base_names = ['Exon','Intron','Intergenic region']
macro_names = ['Macro F-score']
block_names = ['Exon block','Intron block','Gene','Chained intron']
site_boundary_names = ['Transcription start site','Cleavage site']
site_splicing_names = ['Splicing donor site','Splicing acceptor site']
distance_boundary_names = ['Transcription start site','Cleavage site']
distance_splicing_names = ['Splicing donor site','Splicing acceptor site']
    
augustus = read('augustus_test_summary.csv')
augustus['Source']='Augustus'
dl = read('dl_test_summary.csv')
dl['Source']='Deep learning'
revised_dl = read('revised_dl_test_summary.csv')
revised_dl['Source']='Revised deep learning'

data = pd.concat([augustus,dl],sort=True)
save_boxplots(data,'compare/dl_to_augustus')

data = pd.concat([augustus,revised_dl],sort=True)
save_boxplots(data,'compare/revised_to_augustus')

data = pd.concat([dl,revised_dl],sort=True)
save_boxplots(data,'compare/revised_to_raw_dl')