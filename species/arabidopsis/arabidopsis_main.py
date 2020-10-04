import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import read_json, create_folder
from sequence_annotation.preprocess.pipeline import main as pipeline_main
from sequence_annotation.preprocess.select_data import select_ds_data,select_ss_data
from sequence_annotation.preprocess.select_site_region import main as select_site_region_main
from species.arabidopsis.prepair_data import main as prepair_data_main


def main(data_root,upstream_dist,downstream_dist,
         source_name,output_root,kwargs_json_path=None):
    real_protein_gene_id_path = os.path.join(data_root,"raw_data","gene","non_hypothetical_protein_gene_id_chr_1_to_5.txt")
    preprocessed_root = os.path.join(output_root,"preprocessed")
    coordinate_redefined_root=os.path.join(output_root,"coordinate_redefined")
    result_root = os.path.join(output_root,"processed","result")
    output_data_root = os.path.join(output_root,"data")
    half_data_root = os.path.join(output_data_root,"half_data")
    full_data_root = os.path.join(output_data_root,"full_data")
    id_table_path = os.path.join(preprocessed_root,"id_convert.tsv")
    preserved_bed_path = os.path.join(preprocessed_root,"consistent.bed")
    background_bed_path = os.path.join(preprocessed_root,"preprocessed.bed")
    tss_path = os.path.join(preprocessed_root,"tss.gff3")
    cs_path = os.path.join(preprocessed_root,"cleavage_site.gff3")
    genome_path = os.path.join(preprocessed_root,"araport_11_Arabidopsis_thaliana_Col-0_rename.fasta")
    split_root = os.path.join(output_root,"split")
    tss_gff_path = os.path.join(coordinate_redefined_root,"TSS.gff3")
    cs_gff_path = os.path.join(coordinate_redefined_root,"CS.gff3")
    site_region_root = os.path.join(output_root,"site_region")
    score_threshold = None
    create_folder(output_data_root)
    if not os.path.exists(preserved_bed_path):
        prepair_data_main(data_root,preprocessed_root)
    
    kwargs_json = read_json(kwargs_json_path)
    if kwargs_json['use_non_hypothetical_protein_gene_id']:
        kwargs_json['filter_kwargs']['non_hypothetical_protein_gene_id_path'] = real_protein_gene_id_path
        
    if 'score_threshold' in kwargs_json:
        score_threshold = kwargs_json['score_threshold']

    pipeline_main(preserved_bed_path,background_bed_path,id_table_path,genome_path,
                  upstream_dist,downstream_dist,source_name,output_root,
                  tss_path=tss_path,cs_path=cs_path,kwargs_json=kwargs_json)

    names = read_json(os.path.join(split_root,"name_list.json"))
    select_ds_data(result_root,split_root,id_table_path,names,full_data_root)
    select_ss_data(result_root,split_root,id_table_path,names,full_data_root)
    select_ss_data(result_root,split_root,id_table_path,['train_1_2_plus_3_5'],
                   half_data_root,ratio=0.5,select_each_type=True)
    select_ss_data(result_root,split_root,id_table_path,['val_2_minus'],
                   half_data_root,ratio=0.5,select_each_type=True)
    
    select_site_region_main(background_bed_path,id_table_path,genome_path,
                            max(upstream_dist,downstream_dist),site_region_root,
                            treat_strand_independent=True,tss_gff_path=tss_gff_path,
                            cs_gff_path=cs_gff_path,score_threshold=score_threshold,
                            non_hypothetical_protein_gene_id_path=real_protein_gene_id_path)
    
if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--data_root",help="Directory of Arabidopsis thaliana data",required=True)
    parser.add_argument("-o", "--output_root",help="Directory of output folder",required=True)
    parser.add_argument("-u", "--upstream_dist", type=int,help="Upstream distance",required=True)
    parser.add_argument("-d", "--downstream_dist", type=int,help="Downstream distance",required=True)
    parser.add_argument("-s","--source_name",help='Source name',required=True)
    parser.add_argument("-k","--kwargs_json_path")
    args = parser.parse_args()
    main(**vars(args))
