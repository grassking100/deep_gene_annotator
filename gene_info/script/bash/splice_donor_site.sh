bed_file=$1
radius=$2
bed_file="${bed_file%.*}"
awk -F'\t' -v OFS="\t"  '
                         {   
                             n = split($11, sizes, ",")
                             split($12, related_starts, ",")
                             for (i = 1; i <= n;i++)
                             {   
                                 size = sizes[i]
                                 related_start = related_starts[i]
                                 related_end = related_starts[i] + size - 1
                                 if($6=="-")
                                 {
                                     if(related_start>0)
                                     {
                                         splice_donor_site = related_start
                                     }
                                 }
                                 else
                                 {
                                     if((related_end+$2) < ($3-1))
                                     {
                                         splice_donor_site = related_end       
                                     }
                                 }
                                 print($1,splice_donor_site+$2-'$radius',splice_donor_site+$2+'$radius'+1,$4,$5,$6)
                                 
                             }
                         }'  "$bed_file.bed" > "${bed_file}_splice_donor_site_with_radius_${radius}.bed"
                         
                         