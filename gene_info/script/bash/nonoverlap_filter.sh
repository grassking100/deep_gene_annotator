script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
coordinate_file=${1%.*}
#find nonoverlap gene
bash $script_root/sort_merge.sh $coordinate_file.bed

awk -F '\t' -v OFS='\t' '{
    n = split($4, ids, ",")
    if(n==1)
    {
        print(ids[1])
    }
}' ${coordinate_file}_merged.bed > ${coordinate_file}_nonoverlap_id.txt

rm ${coordinate_file}_merged.bed