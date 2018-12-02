selected_bed="${1%.*}"
genome=$2
left=$3
right=$4
bedtools slop -s -i "${selected_bed}.bed" -g $genome -l $left -r $right > "${selected_bed}_merged_expand_left_${left}_right_${right}.bed"
bash script/bash/sort_merge.sh "${selected_bed}_merged_expand_left_${left}_right_${right}.bed"
