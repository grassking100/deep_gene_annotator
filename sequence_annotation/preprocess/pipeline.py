import os
import sys
import pandas as pd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_json
from sequence_annotation.utils.utils import write_json, create_folder
from sequence_annotation.file_process.utils import read_bed, write_bed
from sequence_annotation.file_process.gff_analysis import main as gff_analysis_main
from sequence_annotation.preprocess.redefine_boundary import main as redefine_boundary_main
from sequence_annotation.preprocess.filter import main as filter_main
from sequence_annotation.preprocess.process_data import main as process_data_main
from sequence_annotation.preprocess.split import main as split_main

def main(preserved_bed_path,background_bed_path,id_table_path,tss_path,cs_path,genome_path,
         upstream_dist,downstream_dist,source_name,output_root,kwargs_json=None):
    kwargs = locals()
    kwargs_json = kwargs_json or {}
    redefine_boundary_kwargs = filter_kwargs = process_data_kwargs = {}
    if 'redefine_boundary_kwargs' in kwargs_json:
        redefine_boundary_kwargs = kwargs_json['redefine_boundary_kwargs']
    if 'filter_kwargs' in kwargs_json:
        filter_kwargs = kwargs_json['filter_kwargs']
    if 'process_data_kwargs' in kwargs_json:
        process_data_kwargs = kwargs_json['process_data_kwargs']
    #Set path
    kwarg_path = os.path.join(output_root,"pipeline_kwargs.csv")
    stats_root=os.path.join(output_root,"stats")
    split_root=os.path.join(output_root,"split")
    coordinate_redefined_root = os.path.join(output_root,"coordinate_redefined")
    filtered_root = os.path.join(output_root,"filtered")
    processed_root = os.path.join(output_root,"processed")
    result_root=os.path.join(processed_root,"result")
    ds_root=os.path.join(result_root,"double_strand")
    ss_root=os.path.join(result_root,"single_strand")
    splitted_root=os.path.join(output_root,"split")
    ds_splitted_root=os.path.join(splitted_root,"double_strand")
    ss_splitted_root=os.path.join(splitted_root,"single_strand")
    coordinate_redefined_bed_path = os.path.join(coordinate_redefined_root,"coordinate_redefined.bed")
    filtered_bed_path = os.path.join(filtered_root,"filtered.bed")
    background_redefined_bed_path = os.path.join(processed_root,"background_and_coordinate_redefined.bed")
    region_table_path=os.path.join(result_root,"region_table.tsv")
    ds_rna_gff_path=os.path.join(ds_root,"ds_rna.gff3")
    ds_canonical_gff_path=os.path.join(ds_root,"ds_canonical.gff3")
    ds_region_fasta_path=os.path.join(ds_root,"ds_region.fasta")
    ss_rna_gff_path=os.path.join(ss_root,"ss_rna.gff3")
    ss_canonical_gff_path=os.path.join(ss_root,"ss_canonical.gff3")
    ss_region_fasta_path=os.path.join(ss_root,"ss_region.fasta")
    stats_path = os.path.join(processed_root,"count.json")
    #Write kwargs
    write_json(kwargs,kwarg_path)
    
    #Step 1: Redefining coordinate of transcript
    #if True:
    if not os.path.exists(coordinate_redefined_bed_path):
        redefine_boundary_main(preserved_bed_path,id_table_path,tss_path,cs_path,
                               upstream_dist,downstream_dist,coordinate_redefined_root,**redefine_boundary_kwargs)
    #Step 2: Filtering transcript
    #if True:
    if not os.path.exists(filtered_bed_path):
        filter_main(coordinate_redefined_bed_path,id_table_path,filtered_root,**filter_kwargs)
    #Step 3: Processing data
    if not os.path.exists(stats_path):
        create_folder(processed_root)
        background_bed = read_bed(background_bed_path)
        coordinate_redefined_bed = read_bed(coordinate_redefined_bed_path)
        background_redefined_bed = pd.concat([background_bed,coordinate_redefined_bed])
        write_bed(background_redefined_bed,background_redefined_bed_path)
        process_data_main(filtered_bed_path,background_redefined_bed_path,id_table_path,genome_path,
                          upstream_dist,downstream_dist,source_name,processed_root,**process_data_kwargs)

    #Step 4: Write statistic data of GFF
    create_folder(splitted_root)
    gff_paths = [ds_rna_gff_path,ds_canonical_gff_path,ss_rna_gff_path,ss_canonical_gff_path]
    fasta_paths = [ds_region_fasta_path,ds_region_fasta_path,ss_region_fasta_path,ss_region_fasta_path]
    output_names = ["ds_rna_stats","ds_canonical_stats","ss_rna_stats","ss_canonical_stats"]
    id_sources = ['ordinal_id_wo_strand','ordinal_id_wo_strand','ordinal_id_with_strand','ordinal_id_with_strand']
    for gff_path,fasta_path,output_name,id_source in zip(gff_paths,fasta_paths,output_names,id_sources):
        root = os.path.join(stats_root,output_name)
        gff_analysis_main(gff_path,fasta_path,root,id_source,region_table_path)

    split_main(genome_path+".fai",id_table_path,processed_root,ds_splitted_root,on_double_strand_data=True)
    split_main(genome_path+".fai",id_table_path,processed_root,ss_splitted_root,treat_strand_independent=True)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-p", "--preserved_bed_path",help="Path of preserved transcript data in BED format",required=True)
    parser.add_argument("-b", "--background_bed_path",help="Path of background BED file",required=True)
    parser.add_argument("-t", "--tss_path",help='TSS gff path',required=True)
    parser.add_argument("-c", "--cs_path",help='Cleavage site gff path',required=True)
    parser.add_argument("-g", "--genome_path",help="Path of genome fasta",required=True)
    parser.add_argument("-i", "--id_table_path",help='Transcript and gene id conversion table',required=True)
    parser.add_argument("-o", "--output_root",help="Directory of output folder",required=True)
    parser.add_argument("-u", "--upstream_dist", type=int,help="Upstream distance",required=True)
    parser.add_argument("-d", "--downstream_dist", type=int,help="Downstream distance",required=True)
    parser.add_argument("-s","--source_name",help='Source name',required=True)
    parser.add_argument("-k","--kwargs_json_path")
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs_json = read_json(args.kwargs_json_path)
    main(**kwargs,kwargs_json=kwargs_json)

