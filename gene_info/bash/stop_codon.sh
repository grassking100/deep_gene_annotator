intput_path=$1

awk -F'\t' -v OFS="\t"  '
                         {   
                             if($7 != $8)
                             {
                                 n = split($11, sizes, ",")
                                 split($12, related_starts, ",")
                                 transcrtipt_start = $2 + 1
                                 if($6=="+")
                                 {
                                     stop_codon_stop = $8
                                 }
                                 else
                                 {
                                     stop_codon_stop = $7 + 1
                                 }
                                 
                                 if($6=="+")
                                 {
                                     len=0
                                     for (i = 1; i <= n;i++)
                                     {   
                                         size = sizes[i]
                                         start = related_starts[i] + transcrtipt_start
                                         end = start + size - 1
                                         if(start<=stop_codon_stop && stop_codon_stop<=end)
                                         {
                                             len = len + stop_codon_stop - start
                                             break
                                         }
                                         len = len + size
                                     }
                                 }
                                 else
                                 {
                                     len=0
                                     for (i = n; i > 0;i--)
                                     {   
                                         size = sizes[i]
                                         start = related_starts[i] + transcrtipt_start
                                         end = start + size - 1
                                         if(start<=stop_codon_stop && stop_codon_stop<=end)
                                         {
                                             len = len + end - stop_codon_stop
                                             break
                                         }
                                         len = len + size
                                     }
                                 }
                                 if(len==0)
                                 {
                                     print("Error")
                                 }
                                 print($4,len-2,len+1,$4,$5,$6)
                             }
                         }'  $intput_path
