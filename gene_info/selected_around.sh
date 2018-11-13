bed_file=$1
#order: from 5' to 3'
dist_to_5=$2
dist_to_3=$3
bed_file="${bed_file%.*}"
awk -F'\t' -v OFS="\t"  '
                         {   
                             related_start = $2
                             related_end = $3 - 1
                             if($6=="+")
                             {
                                 print($1,related_start+$2-'$dist_to_5',related_end+$2+'$dist_to_3'+1,$4,$5,$6)
                             }
                             else
                             {
                                 print($1,related_end+$2-'$radius',related_end+$2+'$radius'+1,$4,$5,$6)
                             }
                         }'  "$bed_file.bed" > "${bed_file}_dist_to_five_${dist_to_5}_dist_to_three_${dist_to_3}.bed"
                         
                         