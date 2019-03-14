if (( $# != 2 )); then
    echo "Usage:"
    echo "    bash transcription_start_site.sh BEDFILES RADIUS"
    exit 1
fi
bed_file=${1%.*}
radius=$2
awk -F'\t' -v OFS="\t"  '
                         {   
                             related_start = $2
                             related_end = $3
                             if($6=="+")
                             {
                                 if(related_start-'$radius'>=0)
                                 {
                                     print($1,related_start-'$radius',related_start+'$radius'+1,$4,$5,$6)
                                 }
                             }
                             else
                             {
                                 if(related_end-'$radius'>=0)
                                 {
                                     print($1,related_end-'$radius',related_end+'$radius'+1,$4,$5,$6)
                                 }
                             }
                         }'  "$bed_file.bed" > "${bed_file}_transcription_start_site_with_radius_${radius}.bed"
                         
exit 0