unselected_bed="${1%.*}"
selected_bed="${2%.*}"
genome=$3
left=$4
right=$5
#merge all unselected mRNA
bash sort_merge.sh "$unselected_bed.bed"
#Expand around selected mRNA
bedtools slop -s -i "${selected_bed}.bed" -g $genome -l $left -r $right > "${selected_bed}_expand_left_${left}_right_${right}.bed"
#Output safe zone
bedtools intersect -s -a "${selected_bed}_expand_left_${left}_right_${right}.bed" -b "${unselected_bed}_merged.bed" -wa -v> "${selected_bed}_expand_left_${left}_right_${right}_safe_zone.bed"

#rm "${unselected_bed}_merged_temp.bed"
#rm "${unselected_bed}_sorted.bed"