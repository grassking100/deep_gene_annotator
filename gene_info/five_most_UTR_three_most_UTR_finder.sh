five_utr_file="$1"
five_utr_file="${five_utr_file%.*}"

three_utr_file=$2
three_utr_file="${three_utr_file%.*}"

sort -k4,4  ${five_utr_file}.bed |bedtools groupby -g 4 -c 1,2,3,5,6,1 -o distinct,collapse,collapse,distinct,distinct,count > ${five_utr_file}_merged.bed

sort -k4,4 ${three_utr_file}.bed  |bedtools groupby -g 4 -c 1,2,3,5,6,1 -o distinct,collapse,collapse,distinct,distinct,count > ${three_utr_file}_merged.bed

awk -F'\t' -v OFS="\t"  '   function max_index(array, n)
                            {
                                MAX = array[1]
                                index_ = 1
                                for(i=2;i<=n;i++)
                                {
                                    if(array[i]>MAX)
                                    {
                                        MAX=array[i]
                                        index_=i
                                    }
                                }
                                 return index_
                            }
                            function min_index (array,n)
                            {
                                MIN=array[1]
                                index_=1
                                for(i=2;i<=n;i++)
                                {
                                    if(array[i]<MIN)
                                    {
                                        MIN=array[i]
                                        index_=i
                                    }
                                }
                                return index_
                            };
                            {   
                                 n = $7
                                 split($3, starts, ",")
                                 split($4, ends, ",")
                                 strand=$6
                                 if(strand=="+")
                                 {
                                     five_site = starts[min_index(starts,n)] 
                                     three_site = ends[min_index(starts,n)]
                                     print($2,five_site,three_site,$1,$5,$6)
                                 }
                                 else
                                 {
                                     five_site = ends[max_index(ends,n)]
                                     three_site = starts[max_index(ends,n)] 
                                     print($2,three_site,five_site,$1,$5,$6)
                                 }
                            }' "${five_utr_file}_merged.bed" > "${five_utr_file}_five_most_UTR.bed"
                         
awk -F'\t' -v OFS="\t"  '   function max_index(array, n)
                            {
                                MAX = array[1]
                                index_ = 1
                                for(i=2;i<=n;i++)
                                {
                                    if(array[i]>MAX)
                                    {
                                        MAX=array[i]
                                        index_=i
                                    }
                                }
                                 return index_
                            }
                            function min_index (array,n)
                            {
                                MIN=array[1]
                                index_=1
                                for(i=2;i<=n;i++)
                                {
                                    if(array[i]<MIN)
                                    {
                                        MIN=array[i]
                                        index_=i
                                    }
                                }
                                return index_
                            };
                            {   
                                 n = $7
                                 split($3, starts, ",")
                                 split($4, ends, ",")
                                 strand=$6
                                 if(strand=="+")
                                 {
                                     five_site = starts[max_index(ends,n)] 
                                     three_site = ends[max_index(ends,n)]
                                     print($2,five_site,three_site,$1,$5,$6)
                                 }
                                 else
                                 {
                                     five_site = ends[min_index(starts,n)]
                                     three_site = starts[min_index(starts,n)] 
                                     print($2,three_site,five_site,$1,$5,$6)
                                 }
                            }' "${three_utr_file}_merged.bed" > "${three_utr_file}_three_most_UTR.bed"                         
                         

                         
                         
