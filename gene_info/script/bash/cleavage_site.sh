if (( $# != 2 )); then
    echo "Usage:"
    echo "    bash cleavage_site.sh BEDFILES RADIUS"
    exit 1
fi
bed_file="${1%.*}"
radius=$2
awk -F'\t' -v OFS="\t"  '
                         {   
                             related_start = $2 + 1
                             related_end = $3
                             if($6=="-")
                             {
                                 if(related_start-'$radius'>=1)
                                 {
                                     print($1,related_start-'$radius'-1,related_start+'$radius',$4,$5,$6)
                                 }
                             }
                             else
                             {
                                 if(related_end-'$radius'>=1)
                                 {
                                     print($1,related_end-'$radius'-1,related_end+'$radius',$4,$5,$6)
                                 }
                             }
                         }'  "$bed_file.bed" > "${bed_file}_cleavage_site_with_radius_${radius}.bed"
exit 0             