if (( $# != 2 )); then
    echo "Usage:"
    echo "    bash splice_accept_site.sh BEDFILES RADIUS"
    exit 1
fi
bed_file="${1%.*}"
radius=$2

awk -F'\t' -v OFS="\t"  '
                         {   
                             n = split($11, sizes, ",")
                             transcrtipt_start = $2 + 1
                             split($12, related_starts, ",")
                             for (i = 1; i <= n;i++)
                             {   
                                 size = sizes[i]
                                 related_start = related_starts[i]
                                 related_end = related_starts[i] + size - 1
                                 if($6=="+")
                                 {
                                     if(related_start>0)
                                     {
                                         splice_accept_site = related_start
                                     }
                                 }
                                 else
                                 {
                                     if((related_end+$2) < ($3-1))
                                     {
                                         splice_accept_site = related_end
                                     }
                                 }
                                 start = splice_accept_site+transcrtipt_start-'$radius'
                                 end = splice_accept_site+transcrtipt_start+'$radius'
                                 if(start>=0)
                                 {
                                     print($1,start-1,end,$4,$5,$6)
                                 }
                             }
                         }'  "$bed_file.bed" > "${bed_file}_splice_accept_site_with_radius_${radius}.bed"
exit 0                   