import os
import sys
import deepdish as dd
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, write_json, get_subdict,read_json
from sequence_annotation.file_process.utils import read_gff,read_fasta, read_fai, write_fasta
from sequence_annotation.file_process.utils import write_bed,read_bed
from sequence_annotation.file_process.get_id_table import get_id_convert_dict
from sequence_annotation.file_process.gff2bed import simple_gff2bed
from sequence_annotation.file_process.bed2gff import bed2gff
from sequence_annotation.file_process.get_region_around_site import get_region_around_site
from sequence_annotation.file_process.get_region_around_site import get_donor_site_region, get_acceptor_site_region
from sequence_annotation.file_process.get_region_table import get_region_table_from_fai, write_region_table, read_region_table
from sequence_annotation.visual.plot_composition import main as plot_composition_main
from sequence_annotation.preprocess.filter import main as filter_main
from sequence_annotation.preprocess.split import split
from sequence_annotation.preprocess.get_cleaned_code_bed import get_cleaned_code_bed


def getfasta_command(genome_path,bed_path,output_fasta_path):
    command = "bedtools getfasta -s -name -fi {} -bed {} -fo {}".format(genome_path,bed_path,output_fasta_path)
    os.system(command)


def select_site_region(bed_path,id_table_path,genome_path,radius,output_root,
                       score_mode=None,score_threshold=None,
                       tss_gff_path=None,cs_gff_path=None,
                       non_hypothetical_protein_gene_id_path=None):
    
    filter_root = os.path.join(output_root,"filter")
    region_root = os.path.join(output_root,"region")
    tss_filter_root = os.path.join(filter_root,"filter_for_tss")
    cs_filter_root = os.path.join(filter_root,"filter_for_cs")
    splicing_filter_root = os.path.join(filter_root,"filter_for_splicing")
    create_folder(output_root)
    create_folder(filter_root)
    create_folder(tss_filter_root)
    create_folder(cs_filter_root)
    create_folder(splicing_filter_root)
    tss_transcript_id_path = cs_transcript_id_path = None
    tss_transcript_id_name = cs_transcript_id_name = None
    
    id_convert_dict = get_id_convert_dict(id_table_path)
    #Get region for TSS
    if tss_gff_path is not None:
        tss_gff = read_gff(tss_gff_path,valid_features=False,with_attr=True)
        tss_transcript_id_name = 'including TSS evidence' 
        tss_transcript_id_path = os.path.join(tss_filter_root,"tss_transcript_id.txt")
        tss_transcript_ids = list(tss_gff['belonging'])
        pd.DataFrame.from_dict(tss_transcript_ids).to_csv(tss_transcript_id_path,index=False,header=False)
        
    tss_filter_ids = filter_main(bed_path,id_table_path,tss_filter_root,
                                 remove_non_coding=True,is_target_transcript=True,
                                 non_hypothetical_protein_gene_id_path=non_hypothetical_protein_gene_id_path,
                                 extra_id_path=tss_transcript_id_path,extra_name=tss_transcript_id_name,
                                 output_bed_file=tss_gff_path is None)
    if tss_gff_path is not None:
        tss_filtered_gff = tss_gff[tss_gff['belonging'].isin(tss_filter_ids)].copy()
        tss_filtered_gff['parent'] = tss_filtered_gff['belonging']
        tss_region_gff = get_region_around_site(tss_filtered_gff,radius,radius)
    else:
        tss_filtered_gff = bed2gff(read_bed(tss_filter_path),id_convert_dict)
        tss_region_gff = get_tss_region(tss_filtered_gff,radius,radius)
    
    #Get region for CS
    if cs_gff_path is not None:
        cs_gff = read_gff(cs_gff_path,valid_features=False,with_attr=True)
        cs_transcript_id_name = 'including CS evidence' 
        cs_transcript_id_path = os.path.join(cs_filter_root,"cs_transcript_id.txt")
        cs_transcript_ids = list(cs_gff['belonging'])
        pd.DataFrame.from_dict(cs_transcript_ids).to_csv(cs_transcript_id_path,index=False,header=False)
    cs_filter_ids = filter_main(bed_path,id_table_path,cs_filter_root,
                                remove_non_coding=True,is_target_transcript=True,
                                non_hypothetical_protein_gene_id_path=non_hypothetical_protein_gene_id_path,
                                extra_id_path=cs_transcript_id_path,extra_name=cs_transcript_id_name,
                                output_bed_file=cs_gff_path is None)
    if cs_gff_path is not None:
        cs_filtered_gff = cs_gff[cs_gff['belonging'].isin(cs_filter_ids)].copy()
        cs_filtered_gff['parent'] = cs_filtered_gff['belonging']
        cs_region_gff = get_region_around_site(cs_filtered_gff,radius,radius)
    else:
        cs_filtered_gff = bed2gff(read_bed(cs_filter_path),id_convert_dict)
        cs_region_gff = get_cleavage_site_region(cs_filtered_gff,radius,radius)
    
    #Get region for splicing site    
    filter_main(bed_path,id_table_path,splicing_filter_root,remove_alt_site=True,
                remove_non_coding=True,is_target_transcript=True,
                non_hypothetical_protein_gene_id_path=non_hypothetical_protein_gene_id_path,
                score_threshold=score_threshold,score_mode=score_mode)
    splicing_filtered_path = os.path.join(splicing_filter_root,"filtered.bed")
    splicing_filtered_gff = bed2gff(read_bed(splicing_filtered_path),id_convert_dict)
    ds_region_gff = get_donor_site_region(splicing_filtered_gff,radius,radius)
    as_region_gff = get_acceptor_site_region(splicing_filtered_gff,radius,radius)

    seqs = {}
    labels = {}
    stats = {}
    region_gffs = [tss_region_gff,cs_region_gff,ds_region_gff,as_region_gff]
    site_names = ["TSS","cleavage site","splicing donor site","splicing acceptor site"]
    for type_,region_gff,site_name in zip(['tss','cs','ds','as'],region_gffs,site_names):
        root = os.path.join(region_root,type_)
        create_folder(root)
        origin_region_bed_path = os.path.join(root,'{}_origin_region.bed'.format(type_))
        origin_region_fasta_path = os.path.join(root,'{}_origin_region.fasta'.format(type_))
        region_bed_path = os.path.join(root,'{}_region.bed'.format(type_))
        region_fasta_path = os.path.join(root,'{}_region.fasta'.format(type_))
        composition_path = os.path.join(root,'{}_region.png'.format(type_))
        origin_region_bed = simple_gff2bed(region_gff)
        origin_region_bed = origin_region_bed[~origin_region_bed[['chr','start','end','strand']].duplicated()]
        write_bed(origin_region_bed,origin_region_bed_path)
        getfasta_command(genome_path,origin_region_bed_path,origin_region_fasta_path)
        origin_region_fasta = read_fasta(origin_region_fasta_path)
        cleaned_bed = get_cleaned_code_bed(origin_region_bed, origin_region_fasta)[0]
        write_bed(cleaned_bed,region_bed_path)
        getfasta_command(genome_path,region_bed_path,region_fasta_path)
        region_fasta = read_fasta(region_fasta_path)
        title = "Nucleotide composition around {}".format(site_name)
        plot_composition_main(region_fasta_path, composition_path,shift=-radius,title=title)
        fasta = read_fasta(region_fasta_path)
        stats["{}_number".format(type_)] = {
            'cleaned region':len(region_fasta),
            "origin region":len(origin_region_fasta)
        }
        for id_,seq in fasta.items():
            id_ = "{}_{}".format(type_,id_)
            seqs[id_] = seq
            labels[id_] = type_

    data = (seqs,labels)
    dd.io.save(os.path.join(region_root,'site.h5'),data)
    write_json(stats,os.path.join(region_root,'site_stats.json'))
    
    
def split_site_region(fai_path,region_root,output_root,radius,fold_num=None,treat_strand_independent=False):
    split_root = os.path.join(output_root,'split')
    data_root = os.path.join(output_root,'data')
    region_table_path = os.path.join(split_root,'region_table.tsv')
    create_folder(split_root)
    create_folder(data_root)
    region_table = get_region_table_from_fai(read_fai(fai_path))
    write_region_table(region_table,region_table_path)
    split(fai_path,split_root,fold_num=fold_num,
          region_table_path=region_table_path,
          treat_strand_independent=treat_strand_independent)
    seqs,labels = dd.io.load(os.path.join(region_root,'site.h5'))
    names = read_json(os.path.join(split_root,"name_list.json"))
    ids = {}
    for name in names:
        ids[name] = []
        
    site_names = ["TSS","cleavage site","splicing donor site","splicing acceptor site"]
    for type_,site_name in zip(['tss','cs','ds','as'],site_names):
        region_subroot = os.path.join(region_root,type_)
        fasta_path = os.path.join(region_subroot,"{}_region.fasta".format(type_))
        bed_path = os.path.join(region_subroot,"{}_region.bed".format(type_))
        fasta = read_fasta(fasta_path)
        bed = read_bed(bed_path)
        bed['chr_strand'] = bed['chr']+ "_" +bed['strand']
        for name in names:
            root = os.path.join(data_root,name)
            subroot = os.path.join(root,type_)
            create_folder(root)
            create_folder(subroot)
            subregion_table_path = os.path.join(split_root,name+"_region_table.tsv")
            subregion_table = read_region_table(subregion_table_path)
            chr_strand_ids = set(subregion_table['chr'] + "_" + subregion_table['strand'])
            subbed = bed[bed['chr_strand'].isin(chr_strand_ids)]
            subids = list(subbed['id'])
            ids[name] += [type_+"_"+id_ for id_ in subids]
            subfasta = get_subdict(fasta,subids)
            part_bed_path = os.path.join(subroot,"region.bed")
            part_fasta_path = os.path.join(subroot,"region.fasta")
            composition_path = os.path.join(subroot,"region.png")
            write_bed(subbed,part_bed_path)
            write_fasta(subfasta,part_fasta_path)
            title = "Nucleotide composition around {}".format(site_name)
            plot_composition_main(part_fasta_path, composition_path,shift=-radius,title=title)


    for name in names:
        subseqs = get_subdict(seqs,ids[name])
        sublabels = get_subdict(labels,ids[name])
        subdata = (subseqs,sublabels)
        root = os.path.join(data_root,name)
        dd.io.save(os.path.join(root,"data.h5"),subdata)


def main(bed_path,id_table_path,genome_path,radius,output_root,
         treat_strand_independent=False,fold_num=None,**kwargs):
    fai_path = genome_path+".fai"
    preprocess_root = os.path.join(output_root,'preprocess')
    region_root = os.path.join(preprocess_root,'region')
    process_root = os.path.join(output_root,'process')
    select_site_region(bed_path,id_table_path,genome_path,radius,preprocess_root,**kwargs)
    split_site_region(fai_path,region_root,process_root,radius,fold_num=fold_num,
                      treat_strand_independent=treat_strand_independent)

    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--bed_path",help='Transcript bed path',required=True)
    parser.add_argument("-o", "--output_root",help="Directory of output folder",required=True)
    parser.add_argument("-r", "--radius", type=int,help="Upstream radius",required=True)
    parser.add_argument("-t","--id_table_path",required=True)
    parser.add_argument("-g","--genome_path",required=True)
    parser.add_argument("--score_threshold",type=float,default=None)
    parser.add_argument("--score_mode",type=str,default=None)
    parser.add_argument("--non_hypothetical_protein_gene_id_path")
    parser.add_argument("--tss_gff_path")
    parser.add_argument("--cs_gff_path")
    parser.add_argument("--treat_strand_independent",action='store_true')
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
