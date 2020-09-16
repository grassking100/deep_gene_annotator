import os
import sys
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import write_json, create_folder
from sequence_annotation.file_process.gff2bed import main as gff2bed_main
from sequence_annotation.file_process.utils import create_fai,rename_fasta
from sequence_annotation.file_process.get_id_table import main as get_id_table_main
from sequence_annotation.preprocess.get_consistent_gff import main as get_consistent_gff_main
from species.arabidopsis.preprocess_gff import main as preprocess_gff_main
from species.arabidopsis.preprocess_raw_data import main as preprocess_raw_data_main
from species.arabidopsis.get_non_hypothetical_protein_gene_id import main as get_non_hypothetical_protein_gene_id_main


def main(data_root,output_root):
    kwargs = locals()
    create_folder(output_root)
    kwargs_path = os.path.join(output_root,"arabidopsis_data_prepair_kwargs.csv")
    
    raw_data_root = os.path.join(data_root,"raw_data")
    gro_root = os.path.join(raw_data_root,"homer_tss")
    gro_1 = os.path.join(gro_root,"tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv")
    gro_2 = os.path.join(gro_root,"tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv")
    pac_path = os.path.join(raw_data_root,"PlantAPAdb_pac","arabidopsis_thaliana.Seedling_Control_(SRP089899).all.PAC.bed")
    official_gff_path = os.path.join(raw_data_root,"gene","Araport11_GFF3_genes_transposons.201606.gff")
    raw_genome_path = os.path.join(raw_data_root,"genome","araport_11_Arabidopsis_thaliana_Col-0.fasta")
    genome_path=os.path.join(output_root,"araport_11_Arabidopsis_thaliana_Col-0_rename.fasta")
    id_convert_table_path=os.path.join(output_root,"id_convert.tsv")
    preprocessed_gff_path=os.path.join(output_root,"preprocessed.gff3")
    preprocessed_bed_path=os.path.join(output_root,"preprocessed.bed")    
    consistent_gff_path=os.path.join(output_root,"consistent_official.gff3")
    consistent_bed_path=os.path.join(output_root,"consistent.bed")
    real_protein_gene_id_path = os.path.join(output_root,"non_hypothetical_protein_gene_id_chr_1_to_5.txt")
    valid_chroms = ['1','2','3','4','5']
    write_json(kwargs,kwargs_path)
    #Step 1: Preprocess raw data
    rename_fasta(raw_genome_path,genome_path)
    create_fai(genome_path)
    preprocess_raw_data_main(gro_1,gro_2,pac_path,output_root)
    preprocess_gff_main(official_gff_path,preprocessed_gff_path,["Chr{}".format(chrom) for chrom in valid_chroms])
    get_id_table_main(preprocessed_gff_path,id_convert_table_path)
    gff2bed_main(preprocessed_gff_path,preprocessed_bed_path)
    get_non_hypothetical_protein_gene_id_main(preprocessed_gff_path,real_protein_gene_id_path,
                                              valid_chroms=valid_chroms)
    #Step 2: Get consistence GFF file
    get_consistent_gff_main(preprocessed_gff_path,output_root,'official')
    gff2bed_main(consistent_gff_path,consistent_bed_path)

if __name__ == "__main__":
    #Reading arguments
    parser = ArgumentParser()
    parser.add_argument("-i", "--data_root",help="Directory of Arabidopsis thaliana data",required=True)
    parser.add_argument("-o", "--output_root",help="Directory of output folder",required=True)
    args = parser.parse_args()
    main(**vars(args))
