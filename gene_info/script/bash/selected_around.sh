if (( $# != 3 )); then
    echo "Usage:"
    echo "    bash selected_around.sh BEDFILES DIST_TO_5 DIST_TO_3"
    exit 1
fi

bed_file="${1%.*}"
dist_to_5=$2
dist_to_3=$3

awk -F'\t' -v OFS="\t"  '{   
                             related_start = $2
                             related_end = $3 - 1
                             if($6=="+")
                             {
                                 print($1,related_start-'$dist_to_5',related_end+'$dist_to_3'+1,$4,$5,$6)
                             }
                             else
                             {
                                 print($1,related_start-'$dist_to_3',related_end+'$dist_to_5'+1,$4,$5,$6)
                             }
                         }'  "$bed_file.bed" > "${bed_file}_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}.bed"
                         
exit 0                     