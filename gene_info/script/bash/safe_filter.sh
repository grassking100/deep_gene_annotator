script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
selected_bed=${1%.*}
unselected_bed=${2%.*}
genome=$3
upstream_dist=$4
downstream_dist=$5
saved_root=$6
expand_path=$saved_root/region_upstream_${upstream_dist}_downstream_${downstream_dist}.bed
saved_path=$saved_root/region_upstream_${upstream_dist}_downstream_${downstream_dist}_safe_zone.bed
echo Expand around ${selected_bed}.bed
bedtools slop -s -i ${selected_bed}.bed -g $genome -l $upstream_dist -r $downstream_dist > $expand_path
#Output safe zone
echo Output safe zone of ${selected_bed}.bed
bedtools intersect -s -a $expand_path -b ${unselected_bed}.bed -wa -v > $saved_path
echo Get id
bash $script_root/get_ids.sh $saved_path

rm $saved_path
rm $expand_path
