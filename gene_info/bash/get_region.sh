script_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
if (( $# != 4 )); then
    echo "Usage:"
    echo "    bash selected_around.sh bed_file genome upstream_dist downstream_dist"
    exit 1
fi
selected_bed=${1%.*}
genome=$2
left=$3
right=$4
bedtools slop -s -i $selected_bed.bed -g $genome -l $left -r $right > ${selected_bed}_upstream_${left}_downstream_${right}.bed
bash $script_root/sort_merge.sh ${selected_bed}_upstream_${left}_downstream_${right}.bed
rm ${selected_bed}_upstream_${left}_downstream_${right}.bed
