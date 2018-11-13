bed_file="$1"
bed_file="${bed_file%.*}"
awk -F'\t' -v OFS="\t"  '
                         {   
                             len = split($4, ids, ",")
                             if(len==1)
                             {
                                 print($1,$2,$3,$4,$5,$6)
                             }
                             else
                             {
                                 for(i=1;i<=len;i++)
                                 {
                                     id = ids[i]
                                     print($1,$2,$3,id,$5,$6)
                                 }
                             }
                         }'  "$bed_file.bed" > "${bed_file}_split.bed"
                         
                         