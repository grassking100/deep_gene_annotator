import os
import sys
from scipy.stats import norm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from sklearn.mixture import GaussianMixture
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder
from sequence_annotation.utils.utils import create_folder
from sequence_annotation.file_process.utils import read_fasta,read_gff, write_bed
from sequence_annotation.file_process.utils import get_gff_with_feature_coord
from sequence_annotation.file_process.utils import get_gff_with_intergenic_region,get_gff_with_intron
from sequence_annotation.file_process.get_region_around_site import get_tss_region,get_cleavage_site_region
from sequence_annotation.file_process.get_region_around_site import get_donor_site_region,get_acceptor_site_region
from sequence_annotation.file_process.utils import INTRON_TYPE,EXON_TYPE,INTERGENIC_REGION_TYPE,GENE_TYPE,TRANSCRIPT_TYPE
from sequence_annotation.file_process.get_region_table import read_region_table
from sequence_annotation.visual.plot_composition import main as plot_composition_main


def signal_analysis(gff_path, output_root, tss_radius, cs_radius, donor_radius,acceptor_radius):
    create_folder(output_root)
    signal_radius = 3
    gff = read_gff(gff_path)
    gff = get_gff_with_intron(gff)
    tss_region = get_tss_region(gff, tss_radius,tss_radius)
    cs_region = get_cleavage_site_region(gff, cs_radius, cs_radius)
    donor_region = get_donor_site_region(gff, donor_radius, donor_radius)
    acceptor_region = get_acceptor_site_region(gff, acceptor_radius,acceptor_radius)
    tss_signal_region = get_tss_region(gff, signal_radius, signal_radius)
    cs_signal_region = get_cleavage_site_region(gff, signal_radius,signal_radius)
    donor_signal_region = get_donor_site_region(gff, signal_radius, signal_radius)
    acceptor_signal_region = get_acceptor_site_region(gff, signal_radius,signal_radius)
    core_donor_signal_region = get_donor_site_region(gff, 0, 1)
    core_acceptor_signal_region = get_acceptor_site_region(gff, 1, 0)
    regions = [
        tss_region, cs_region, donor_region, acceptor_region,
        tss_signal_region, cs_signal_region, donor_signal_region,acceptor_signal_region,
        core_donor_signal_region,core_acceptor_signal_region
    ]
    names = [
        "tss_around_{}".format(tss_radius),"cleavage_site_around_{}".format(cs_radius),
        "donor_site_around_{}".format(donor_radius),"acceptor_site_around_{}".format(acceptor_radius), 
        "tss_signal","cleavage_site_signal", "donor_site_signal", "acceptor_site_signal",
        "core_donor_site_signal", "core_acceptor_site_signal"
    ]
    for region, name in zip(regions, names):
        write_bed(region, os.path.join(output_root, "{}.bed".format(name)))


def norm_fit_log10(data, component_num=None):
    data = np.log10(data)
    component_num = component_num or 1
    if component_num == 1:
        mean, std = norm.fit(data)
        weights,means,stds = [1], [mean], [std]
    else:
        clf = GaussianMixture(n_components=component_num,covariance_type='spherical',
                              max_iter=1000,random_state=0)
        clf.fit(data.reshape(-1, 1))
        weights = clf.weights_
        means = clf.means_.flatten()
        stds = np.sqrt(clf.covariances_)
    params = {'weights': weights, 'means': means, 'stds': stds}
    return params


def get_norm_pdf(params, range_, merge=True):
    pdfs = []
    for index in range(len(params['weights'])):
        weight = params['weights'][index]
        mean = params['means'][index]
        std = params['stds'][index]
        pdf = norm.pdf(range_, mean, std) * weight
        pdfs.append(pdf)
    pdfs = np.array(pdfs)
    if merge:
        pdf_sum = pdfs.sum(0)
        return pdf_sum
    else:
        return pdfs


def plot_log_gaussian(data, params=None, merge=False, title=None,density=True,show_legend=True):
    plt.clf()
    plt.locator_params(axis='x', nbins=5)
    font_size = 16
    step = 1000
    log_data = np.log10(data)
    range_ = np.arange(min(log_data) * step, max(log_data) * step) / step
    if params is not None:
        pdfs = get_norm_pdf(params, range_, merge)
        plt.plot(range_, pdfs.sum(0), label='model summation')
        for index, pdf in enumerate(pdfs):
            plt.plot(range_, pdf, label='model {}'.format(index + 1))
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.hist(np.log10(data), bins=100, density=density)
    plt.xlabel('log10(length)', fontsize=font_size)
    plt.ylabel('density', fontsize=font_size)
    if title is not None:
        plt.title(title, fontsize=font_size)
    if show_legend:
        plt.legend(fontsize=font_size)


def length_gaussian_modeling(gff_path, output_root, component_num=None):
    create_folder(output_root)
    data_types = {'gene': GENE_TYPE,'intron': INTRON_TYPE,'exon': EXON_TYPE}
    gaussian_model_path = os.path.join(output_root,'length_log10_gaussian_model.tsv')
    gff = read_gff(gff_path)
    gff = get_gff_with_intron(gff)
    gaussian_models = []
    for type_, feature in data_types.items():
        title = 'The distribution and Gaussian model of {}\'s log-length (nt)'.format(type_)
        path = os.path.join(output_root,'{}_log10_length_gaussian_model'.format(type_))
        subgff = gff[gff['feature']==feature]
        lengths = np.array(list(subgff['end'] - subgff['start'] + 1))
        params = norm_fit_log10(lengths, component_num)
        plot_log_gaussian(lengths, params, title=title)
        plt.savefig(path,bbox_inches = 'tight',pad_inches = 0)
        for weight, mean, std in zip(params['weights'], params['means'],params['stds']):
            gaussian_models.append({'weights': weight,'means': mean,'stds': std,'types': type_})
    gaussian_models = pd.DataFrame.from_dict(gaussian_models)[['types', 'weights', 'means', 'stds']]
    gaussian_models.to_csv(gaussian_model_path, sep='\t', index=None)
        

def get_feature_stats(gff,region_table,chrom_source):
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
        
    
def plot_length(length_dict,output_root):
    for feature,lengths in length_dict.items():
        plt.figure()
        plt.hist(lengths,100)
        plt.title("The distribution of \n{}'s lengths (nt)".format(feature), fontsize=14)
        plt.xlabel("length", fontsize=14)
        plt.ylabel("number", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(output_root,'{}_length.png'.format(feature).replace(' ','_')),
                   bbox_inches = 'tight',pad_inches = 0)
        
        
def plot_log_length(length_dict,output_root,predicted_mode=False,is_small_figure=False):
    component_nums = {INTRON_TYPE:2,EXON_TYPE:2,INTERGENIC_REGION_TYPE:3,GENE_TYPE:2,TRANSCRIPT_TYPE:2}
    if predicted_mode:
        for key,value in component_nums.items():
            component_nums[key] += 1
        
    for feature,lengths in length_dict.items():
        if feature == INTERGENIC_REGION_TYPE and not predicted_mode:
            with_gaussian = False
        else:
            with_gaussian = True
        params = None
        title = "The log distribution of \n{}'s lengths (nt)".format(feature)
        if with_gaussian:
            model_path = os.path.join(output_root,"{}_models.tsv").format(feature.replace(' ','_'))
            title = 'The distribution and Gaussian model\n of {}\'s log-length (nt)'.format(feature)
            if feature in component_nums:
                component_num = component_nums[feature]
            else:
                component_num = 1
            params = norm_fit_log10(lengths, component_num)
            pd.DataFrame.from_dict(params).to_csv(model_path,sep='\t')
        if is_small_figure:
            plt.figure(figsize=(5,4))
            plot_log_gaussian(lengths, params,show_legend=False)
        else:
            plot_log_gaussian(lengths, params, title=title,show_legend=with_gaussian)
        plt.savefig(os.path.join(output_root,'{}_log10_length.png'.format(feature).replace(' ','_')),
                    bbox_inches = 'tight',pad_inches = 0)
    
    
def gff_feature_stats(gff_path,region_table_path,chrom_source,output_root,**kwargs):
    create_folder(output_root)
    gff = read_gff(gff_path)
    region_table = read_region_table(region_table_path)
    stats,lengths = get_feature_stats(gff,region_table,chrom_source)
    stats_df = pd.DataFrame.from_dict(stats)
    stats_df.to_csv(os.path.join(output_root,'feature_stats.tsv'),sep='\t',index=None)
    plot_length(lengths,output_root)
    plot_log_length(lengths,output_root,**kwargs)
    

def motif_count(fasta_path, output_path):
    fasta = read_fasta(fasta_path)
    seqs = []
    for id_, seq in fasta.items():
        seqs.append({'id': id_, 'motif': seq})
    seqs = pd.DataFrame.from_dict(seqs)
    counts = seqs['motif'].value_counts().to_frame()
    counts.index.name = 'motif'
    counts.columns = ['count']
    counts.to_csv(output_path, sep='\t', header=True)
    
    
def main(input_gff_path,genome_path,output_root,chrom_source,region_table_path,
         tss_radius=None,cs_radius=None,donor_radius=None,acceptor_radius=None):
    tss_radius = tss_radius or 100
    cs_radius = cs_radius or 100
    donor_radius = donor_radius or 100
    acceptor_radius = acceptor_radius or 100
    bed_root=os.path.join(output_root,"bed")
    signal_stats_root=os.path.join(output_root,"signal_stats")
    core_signal_stats_root=os.path.join(output_root,"core_signal_stats")
    fasta_root=os.path.join(output_root,"fasta")
    composition_root=os.path.join(output_root,"composition")
    feature_stats_root = os.path.join(output_root,"feature_stats")
    length_gaussian_root = os.path.join(output_root,"length_gaussian")
    signal_analysis(input_gff_path,bed_root,tss_radius,cs_radius,donor_radius,acceptor_radius)
    create_folder(fasta_root)
    create_folder(composition_root)
    create_folder(signal_stats_root)
    create_folder(core_signal_stats_root)
    
    for name in os.listdir(bed_root):
        name = name.split('.')
        if name[-1] == 'bed':
            prefix = '.'.join(name[:-1])
            bed_path = os.path.join(bed_root,"{}.bed".format(prefix))
            fasta_path = os.path.join(fasta_root,"{}.fasta".format(prefix))
            command = "bedtools getfasta -s -name -fi {} -bed {} -fo {}".format(genome_path,bed_path,fasta_path)
            os.system(command)
            
    names = ["tss_around_{}".format(tss_radius),"cleavage_site_around_{}".format(cs_radius),
             "donor_site_around_{}".format(donor_radius),"acceptor_site_around_{}".format(acceptor_radius)]
    radiuses = [tss_radius,cs_radius,donor_radius,acceptor_radius]
    site_names = ["TSS","cleavage site","splicing donor site","splicing acceptor site"]
    for name,radius,site_name in zip(names,radiuses,site_names):
        fasta_path = os.path.join(fasta_root,name+".fasta")
        title = "Nucleotide composition around {}".format(site_name)
        figure_path = os.path.join(composition_root,name+".png")
        plot_composition_main(fasta_path,figure_path,shift=radius,title=title)
                              
    for name in ["tss_signal","cleavage_site_signal","donor_site_signal","acceptor_site_signal"]:
        fasta_path = os.path.join(fasta_root, name+".fasta")
        signal_path = os.path.join(signal_stats_root, name+".tsv")
        motif_count(fasta_path,signal_path)

    for name in ["core_donor_site_signal","core_acceptor_site_signal"]:
        fasta_path = os.path.join(fasta_root, name+".fasta")
        signal_path = os.path.join(core_signal_stats_root, name+".tsv")
        motif_count(fasta_path,signal_path)
        
        
    gff_feature_stats(input_gff_path,region_table_path,chrom_source,feature_stats_root)
    length_gaussian_modeling(input_gff_path,length_gaussian_root,2)

    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_gff_path",help="Path of GFF file",required=True)
    parser.add_argument("-g", "--genome_path",help="Path of genome fasta",required=True)
    parser.add_argument("-o", "--output_root",required=True)
    parser.add_argument("-s","--chrom_source",required=True)
    parser.add_argument("-r","--region_table_path",required=True)
    parser.add_argument("-t", "--tss_radius",type=int,default=None)
    parser.add_argument("-c", "--cs_radius",type=int,default=None)
    parser.add_argument("-d", "--donor_radius",type=int,default=None)
    parser.add_argument("-a", "--acceptor_radius",type=int,default=None)
    args = parser.parse_args()
    main(**vars(kwargs))
