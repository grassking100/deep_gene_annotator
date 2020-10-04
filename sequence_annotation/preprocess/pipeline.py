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
from sequence_annotation.preprocess.split import split

def main(preserved_bed_path,background_bed_path,id_table_path,genome_path,
         upstream_dist,downstream_dist,source_name,output_root,
         tss_path=None,cs_path=None,kwargs_json=None):
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

    split_root=os.path.join(output_root,"split")
    coordinate_redefined_root = os.path.join(output_root,"coordinate_redefined")
    filtered_root = os.path.join(output_root,"filtered")
    processed_root = os.path.join(output_root,"processed")
    result_root=os.path.join(processed_root,"result")
    ds_root=os.path.join(result_root,"double_strand")
    ss_root=os.path.join(result_root,"single_strand")
    stats_root=os.path.join(ss_root,"stats")
    splitted_root=os.path.join(output_root,"split")
    filtered_bed_path = os.path.join(filtered_root,"filtered.bed")
    background_redefined_bed_path = os.path.join(processed_root,"background_and_coordinate_redefined.bed")
    region_table_path=os.path.join(result_root,"region_table.tsv")
    ds_rna_gff_path=os.path.join(ds_root,"transcript.gff3")
    ds_canonical_gff_path=os.path.join(ds_root,"canonical_gene.gff3")
    ds_region_fasta_path=os.path.join(ds_root,"region.fasta")
    ss_rna_gff_path=os.path.join(ss_root,"transcript.gff3")
    ss_canonical_gff_path=os.path.join(ss_root,"canonical_gene.gff3")
    ss_region_fasta_path=os.path.join(ss_root,"region.fasta")
    stats_path = os.path.join(processed_root,"count.json")
    splitted_log_path = os.path.join(splitted_root,'split.log')
    #Write kwargs
    write_json(kwargs,kwarg_path)
    
    #Step 1: Redefining coordinate of transcript
    if tss_path is not None and cs_path is not None:
        quality_coordinate_bed_path = os.path.join(coordinate_redefined_root,"coordinate_redefined.bed")
        if not os.path.exists(quality_coordinate_bed_path):
            redefine_boundary_main(preserved_bed_path,id_table_path,tss_path,cs_path,
                                   upstream_dist,downstream_dist,coordinate_redefined_root,
                                   **redefine_boundary_kwargs)
    else:
        quality_coordinate_bed_path = preserved_bed_path
        
    #Step 2: Filtering transcript
    if not os.path.exists(filtered_bed_path):
        filter_main(quality_coordinate_bed_path,id_table_path,filtered_root,**filter_kwargs)

    #Step 3: Processing data
    if not os.path.exists(stats_path):
        create_folder(processed_root)
        background_bed = read_bed(background_bed_path)
        coordinate_redefined_bed = read_bed(quality_coordinate_bed_path)
        background_redefined_bed = pd.concat([background_bed,coordinate_redefined_bed])
        write_bed(background_redefined_bed,background_redefined_bed_path)
        process_data_main(filtered_bed_path,background_redefined_bed_path,id_table_path,genome_path,
                          upstream_dist,downstream_dist,source_name,processed_root,**process_data_kwargs)

    #Step 4: Write statistic data of GFF
    create_folder(splitted_root)
    gff_paths = [ss_rna_gff_path,ss_canonical_gff_path]
    output_names = ["transcript_stats","canonical_gene_stats"]
    
    for gff_path,output_name in zip(gff_paths,output_names):
        root = os.path.join(stats_root,output_name)
        stats_log_path = os.path.join(root,'stats.log')
        if not os.path.exists(stats_log_path):
            gff_analysis_main(gff_path,ss_region_fasta_path,root,'ordinal_id_with_strand',region_table_path)
            with open(stats_log_path,'w') as fp:
                fp.write("Finish")

    if not os.path.exists(splitted_log_path):
        split(genome_path+".fai",splitted_root,region_table_path=region_table_path,treat_strand_independent=True)
        with open(splitted_log_path,'w') as fp:
            fp.write("Finish")
            

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-p", "--preserved_bed_path",help="Path of preserved transcript data in BED format",required=True)
    parser.add_argument("-b", "--background_bed_path",help="Path of background BED file",required=True)
    parser.add_argument("-t", "--tss_path",help="Path of TSS evidence GFF file")
    parser.add_argument("-c", "--cs_path",help="Path of cleavage sites evidence GFF file")
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
