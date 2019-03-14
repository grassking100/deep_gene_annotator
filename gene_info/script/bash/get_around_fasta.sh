#!/bin/bash
bed="${1%.*}"
genome=$2
dist_to_5=$3
dist_to_3=$4
TSS_radius=$5
donor_radius=$6
accept_radius=$7
cleavage_radius=$8

bash script/bash/transcription_start_site.sh ${bed}.bed ${TSS_radius}
bash script/bash/splice_donor_site.sh ${bed}.bed ${donor_radius}
bash script/bash/splice_accept_site.sh ${bed}.bed ${accept_radius}
bash script/bash/cleavage_site.sh ${bed}.bed ${cleavage_radius}
bash script/bash/selected_around.sh ${bed}.bed ${dist_to_5} ${dist_to_3}

Rscript script/R/drop_duplicate_by_simple_coordinate.R ${bed}_transcription_start_site_with_radius_${TSS_radius}.bed
Rscript script/R/drop_duplicate_by_simple_coordinate.R ${bed}_splice_donor_site_with_radius_${donor_radius}.bed
Rscript script/R/drop_duplicate_by_simple_coordinate.R ${bed}_splice_accept_site_with_radius_${accept_radius}.bed
Rscript script/R/drop_duplicate_by_simple_coordinate.R ${bed}_cleavage_site_with_radius_${cleavage_radius}.bed

bedtools getfasta -s -fi ${genome} -bed ${bed}_transcription_start_site_with_radius_${TSS_radius}_unique_simple_coordinate.bed -fo ${bed}_transcription_start_site_with_radius_${TSS_radius}_unique_simple_coordinate.fasta

bedtools getfasta -s -fi ${genome} -bed ${bed}_splice_donor_site_with_radius_${donor_radius}_unique_simple_coordinate.bed -fo ${bed}_splice_donor_site_with_radius_${donor_radius}_unique_simple_coordinate.fasta

bedtools getfasta -s -fi ${genome} -bed ${bed}_splice_accept_site_with_radius_${accept_radius}_unique_simple_coordinate.bed -fo ${bed}_splice_accept_site_with_radius_${accept_radius}_unique_simple_coordinate.fasta

bedtools getfasta -s -fi ${genome} -bed ${bed}_cleavage_site_with_radius_${cleavage_radius}_unique_simple_coordinate.bed -fo ${bed}_cleavage_site_with_radius_${cleavage_radius}_unique_simple_coordinate.fasta

bedtools getfasta -s -fi ${genome} -bed ${bed}.bed -fo ${bed}.fasta

bash script/bash/sort_merge.sh ${bed}_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}.bed

bedtools getfasta -s -name -fi ${genome} -bed ${bed}_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}_merged.bed -fo ${bed}_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}_merged.fasta 

bedtools getfasta -s -fi ${genome} -bed ${bed}.bed -fo ${bed}.fasta 
