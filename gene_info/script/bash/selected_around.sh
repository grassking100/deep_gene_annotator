if (( $# != 3 )); then
    echo "Usage:"
    echo "    bash selected_around.sh bed_file upstream_dist downstream_dist"
    exit 1
fi

bed_file=${1%.*}
upstream_dist=$2
downstream_dist=$3

awk -F'\t' -v OFS="\t"  '{   
                             related_start = $2 + 1
                             related_end = $3
                             if($6=="+")
                             {   if(related_start-'$upstream_dist'>=1)
                                 {
                                     print($1,related_start-'$upstream_dist'-1,related_end+'$downstream_dist',$4,$5,$6)
                                 }
                             }
                             else
                             {
                                 if(related_start-'$downstream_dist'>=1)
                                 {
                                     print($1,related_start-'$downstream_dist'-1,related_end+'$upstream_dist',$4,$5,$6)
                                 }

                             }
                         }'  $bed_file.bed > ${bed_file}_upstream_${upstream_dist}_downstream_${downstream_dist}.bed
                         
exit 0                     