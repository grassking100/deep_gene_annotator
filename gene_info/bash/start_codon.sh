intput_path=$1

awk -F'\t' -v OFS="\t"  '
                         {   
                             if($7 != $8)
                             {
                                 n = split($11, sizes, ",")
                                 transcrtipt_start = $2 + 1
                                 if($6=="+")
                                 {
                                     start_codon_start = $7 + 1
                                 }
                                 else
                                 {
                                     start_codon_start = $8
                                 }
                                 split($12, related_starts, ",")
                                 if($6=="+")
                                 {
                                     len=0
                                     for (i = 1; i <= n;i++)
                                     {   
                                         size = sizes[i]
                                         start = related_starts[i] + transcrtipt_start
                                         end = start + size - 1
                                         if(start<=start_codon_start && start_codon_start<=end)
                                         {
                                             len = len + start_codon_start - start
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
                                         if(start<=start_codon_start && start_codon_start<=end)
                                         {
                                             len = len + end - start_codon_start
                                             break
                                         }
                                         len = len + size
                                     }
                                 }
                                 if(len==0)
                                 {
                                     print("Error")
                                 }
                                 print($4,len,len+3,$4,$5,$6)
                             }
                         }'  $intput_path
