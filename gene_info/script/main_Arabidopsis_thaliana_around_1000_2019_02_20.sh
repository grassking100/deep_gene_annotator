#!/bin/bash
folder_name=2019_02_20
#$(date +"%Y_%m_%d")
root="/home/io/Arabidopsis_thaliana"
saved_root=${root}/data/${folder_name}
coordinate_consist_with_gene_id=${saved_root}/coordinate_consist_with_gene_id
coordinate_consist=${saved_root}/coordinate_consist
separate_path=${saved_root}/separate
want_bed=${separate_path}/want
unwant_bed=${separate_path}/unwant
mRNA_file=${root}/data/Araport11_mRNA_2018_11_27
fai=${root}/data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta.fai
result_path=${saved_root}/result
exon_file=${root}/data/Araport11_exon_2018_11_27
genome_file=${root}/data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
dist_to_5=1000
dist_to_3=1000
TSS_radius=200
donor_radius=200
accept_radius=200
cleavage_radius=200
echo "Start of program"
mkdir ${result_path}
mkdir ${separate_path}
mkdir ${saved_root}

#Load library
Rscript --save script/R/utils.R
Rscript --save --restore script/R/gff.R
Rscript --save --restore script/R/belong_finder.R
Rscript --save --restore script/R/merge_and_clean.R
#Read and run
Rscript --save --restore script/R/read.R ${root}/data/
Rscript --save --restore script/R/find_belong.R ${saved_root}/

#Write to coordinate_consist_with_gene_id.bed
#Write to coordinate_consist.bed
#--Write to pure_coordinate_consist.bed
Rscript --save --restore script/R/coordinate_export.R ${saved_root}/
#Write nonoverlap gene id to coordinate_consist_with_gene_id_nonoverlap_id.txt

bash script/bash/nonoverlap_filter.sh ${coordinate_consist_with_gene_id}.bed
#Read Araport11_mRNA.bed and coordinate_consist_with_gene_id_nonoverlap_id.txt and #${coordinate_consist}.bed
#Write to create mRNA_coordinate.bed
bash script/bash/recurrent_cleaner.sh $coordinate_consist_with_gene_id $coordinate_consist \
$mRNA_file $want_bed $unwant_bed $fai $dist_to_5 $dist_to_3 $separate_path $result_path

#Export to ${exon_file}_merged_with_coordinate_file.bed
bash script/bash/merge_feature_and_mRNA_by_id.sh ${exon_file}.bed mRNA_coordinate.bed ${result_path} exon_merged_with_coordinate_file.bed
Rscript --save --restore script/R/merge_bed.R ${result_path}/exon_merged_with_coordinate_file.bed ${result_path}/result.bed
rm ${result_path}/exon_merged_with_coordinate_file.bed
bash script/bash/get_around_fasta.sh ${result_path}/result.bed ${genome_file} ${dist_to_5} \
${dist_to_3} ${TSS_radius} ${donor_radius} ${accept_radius} ${cleavage_radius}

python3 script/python/create_ann_region.py -m ${result_path}/result.bed \
-r ${result_path}/result_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}_merged.bed \
-f $fai -s $folder_name -o ${result_path}/result_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}_merged.h5
echo "End of program"