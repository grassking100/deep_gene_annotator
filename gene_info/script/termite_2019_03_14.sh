#!/bin/bash
folder_name="isoseq.ccs.polished.hq.fasta_align_to_termite_g1_2019_03_01"
root="/home/io/long_read/termite"
saved_root=${root}/${folder_name}
coordinate_bed=${saved_root}/isoseq.ccs.polished.hq.fasta_align_to_termite_g1_2018_9_7_Ching_Tien_Wang_merged_with_coordinate_file
coordinate_consist=${saved_root}/isoseq.ccs.polished.hq.fasta_align_to_termite_g1_2018_9_7_Ching_Tien_Wang_merged_with_coordinate_file
separate_path=${saved_root}/separate
want_bed=${separate_path}/want
unwant_bed=${separate_path}/unwant
result_path=${saved_root}/result
fai=${root}/raw_data/genome/termite_g1.fasta.fai
genome_file=${root}/raw_data/genome/termite_g1.fasta
mRNA_file=${saved_root}/isoseq.ccs.polished.hq.fasta_align_to_termite_g1_2018_9_7_Ching_Tien_Wang_mRNA
exon_file=${saved_root}/isoseq.ccs.polished.hq.fasta_align_to_termite_g1_2018_9_7_Ching_Tien_Wang_CDS
dist_to_5=500
dist_to_3=500
TSS_radius=200
donor_radius=200
accept_radius=200
cleavage_radius=200
echo "Start of program"
mkdir ${result_path}
mkdir ${separate_path}
mkdir ${saved_root}

bash script/bash/nonoverlap_filter.sh ${coordinate_bed}.bed

bash script/bash/recurrent_cleaner.sh $coordinate_bed $coordinate_bed \
$mRNA_file $want_bed $unwant_bed $fai $dist_to_5 $dist_to_3 $separate_path $result_path

bash script/bash/merge_feature_and_mRNA_by_id.sh ${exon_file}.bed ${mRNA_file}.bed ${result_path} CDS_merged_with_coordinate_file.bed
Rscript --save --restore script/R/merge_bed.R ${result_path}/CDS_merged_with_coordinate_file.bed ${result_path}/result.bed

bash script/bash/get_around_fasta.sh ${result_path}/result.bed ${genome_file} ${dist_to_5} \
${dist_to_3} ${TSS_radius} ${donor_radius} ${accept_radius} ${cleavage_radius}

python3 script/python/create_ann_region.py -m ${result_path}/result.bed \
-r ${result_path}/result_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}_merged.bed \
-f $fai -s $folder_name -o ${result_path}/result_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}_merged.h5
echo "End of program"