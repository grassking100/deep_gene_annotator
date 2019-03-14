coordinate_file="${1%.*}"
#output_file=$2
#find non overlap gene
bash script/bash/sort_merge.sh "${coordinate_file}.bed"

awk -F '\t' -v OFS='\t' '{
    n = split($4, ids, ",")
    if(n==1)
    {
        print(ids[1])
    }
}' "${coordinate_file}_merged.bed" > "${coordinate_file}_nonoverlap_id.txt"