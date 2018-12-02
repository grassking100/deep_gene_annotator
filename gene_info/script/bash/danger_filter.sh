selected_bed="${1%.*}"
unselected_bed="${2%.*}"
genome=$3
left=$4
right=$5
#merge all unselected mRNA
echo Merging "${unselected_bed}.bed"
bash script/bash/sort_merge.sh "${unselected_bed}.bed"
#merge all selected mRNA
echo Merging "${selected_bed}.bed"
bash script/bash/sort_merge.sh "${selected_bed}.bed"
#Expand around selected mRNA
echo Expand around "${selected_bed}.bed"
bedtools slop -s -i "${selected_bed}.bed" -g $genome -l $left -r $right > "${selected_bed}_expand_left_${left}_right_${right}.bed"
#Output safe zone
echo Output safe zone of "${selected_bed}.bed"
bedtools intersect -s -a "${selected_bed}_expand_left_${left}_right_${right}.bed" -b "${unselected_bed}_merged.bed" -wa -v> "${selected_bed}_expand_left_${left}_right_${right}_safe_zone.bed"
echo Get id
bash script/bash/get_ids.sh "${selected_bed}_expand_left_${left}_right_${right}_safe_zone.bed"

rm "${selected_bed}_merged.bed"
rm "${unselected_bed}_merged.bed"