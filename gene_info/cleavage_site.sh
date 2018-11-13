bed_file="$1"
radius=$2
bed_file="${bed_file%.*}"
awk -F'\t' -v OFS="\t"  '
                         {   
                             n = split($11, sizes, ",")
                             split($12, related_starts, ",")
                             for (i = 1; i<= n;i++)
                             {   
                                 size = sizes[i]
                                 related_start = related_starts[i]
                                 related_end = related_starts[i] + size - 1
                                 if($6=="-")
                                 {
                                     if(related_start == 0)
                                     {

                                         print($1,related_start+$2-'$radius',related_start+$2+'$radius'+1,$4,$5,$6)
                                     }
                                 }
                                 else
                                 {
                                     if(related_end+$2 == $3-1)
                                     {
                                         print($1,related_end+$2-'$radius',related_end+$2+'$radius'+1,$4,$5,$6)
                                     }
                                 }

                             }
                         }'  "$bed_file.bed" > "${bed_file}_cleavage_site_with_radius_${radius}.bed"
                         
                         