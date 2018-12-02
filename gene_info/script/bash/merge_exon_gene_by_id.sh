exon_file=$1
coordinate_file=$2
exon_file="${exon_file%.*}"
coordinate_file="${coordinate_file%.*}"

awk -F'\t' -v OFS="\t" '{gsub("\r", "",$0); print $1,$2,$3,$4,$5,$6,".",".","exon"}' ${exon_file}.bed > ${exon_file}.bed_extend
awk -F'\t' -v OFS="\t" '{gsub("\r", "",$0); print $0,".",".","mRNA"}' ${coordinate_file}.bed > ${coordinate_file}.bed_extend
cat ${exon_file}.bed_extend ${coordinate_file}.bed_extend |sort -k4,4 > temp.bed
cat ${exon_file}.bed_extend ${coordinate_file}.bed_extend |sort -k4,4 |bedtools groupby -g 4 -c 1,2,3,5,6,7,8,9 -o distinct,collapse,collapse,distinct,distinct,distinct,distinct,collapse  | awk -F'\t' -v OFS="\t"  '{print $2,$3,$4,$1,$5,$6,$7,$8,$9}' > ${exon_file}_merged_temp.bed

awk -F'\t' -v OFS="\t"  '{
                            n = split($9, types, ",")
                            split($2, starts, ",")
                            split($3, ends, ",")
                            exist_mRNA = 0
                            for(i=1;i<=n;i++)
                            {
                                if(types[i]=="mRNA")
                                {
                                    mRNA_start = starts[i]
                                    mRNA_end = ends[i]
                                    exist_mRNA = 1
                                }
                            }
                            if(exist_mRNA)
                            {
                                exon_index=0
                                for(i=1;i<=n;i++)
                                {
                                    if(types[i]=="exon")
                                    {
                                        start = starts[i]
                                        end = ends[i]
                                        if(mRNA_start > start)
                                        {
                                            start=mRNA_start
                                        }
                                        if(mRNA_end < end)
                                        {
                                            end=mRNA_end
                                        }
                                        exon_index += 1
                                        exon_starts[exon_index] = start
                                        exon_ends[exon_index] = end

                                    }
                                }
                                exon_start_min = exon_starts[1]
                                exon_end_max = exon_ends[1]
                                min_index=1
                                max_index=1
                                for(i=2;i<=exon_index;i++)
                                {
                                    if(exon_start_min>exon_starts[i])
                                    {
                                        exon_start_min=exon_starts[i]
                                        min_index=i
                                    }
                                    if(exon_end_max<exon_ends[i])
                                    {
                                        exon_end_max=exon_ends[i]
                                        max_index=i
                                    }
                                }
                                if(exon_end_max<mRNA_end)
                                {
                                    exon_ends[max_index]=mRNA_end
                                }
                                if(exon_start_min>mRNA_start)
                                {
                                    exon_starts[min_index]=mRNA_start
                                }
                                exon_starts_str = exon_starts[1] - mRNA_start
                                exon_sizes_str = exon_ends[1] - exon_starts[1]
                                for(i=2;i<=exon_index;i++)
                                {
                                    exon_starts_str = exon_starts_str "," exon_starts[i] - mRNA_start
                                    exon_sizes_str = exon_sizes_str "," exon_ends[i]-exon_starts[i]
                                }
                                orf_start_number = split($7, orf_starts, ",")
                                orf_end_number = split($8, orf_ends, ",")
                                orf_start_str = ""
                                orf_end_str = ""
                                for(i=1;i<=orf_start_number;i++)
                                {
                                    if(orf_starts[i]!=".")
                                    {
                                        if(orf_start_str=="")
                                        {
                                            orf_start_str = orf_starts[i]
                                        }
                                        else
                                        {
                                            orf_start_str = orf_start_str "," orf_starts[i]
                                        }                                        
                                    }
                                    
                                }
                                for(i=1;i<=orf_end_number;i++)
                                {
                                    if(orf_ends[i]!=".")
                                    {
                                        if(orf_end_str=="")
                                        {
                                            orf_end_str = orf_ends[i]
                                        }
                                        else
                                        {
                                            orf_end_str = orf_end_str "," orf_ends[i]
                                        }                                        
                                    }
                                    
                                }
                                if(orf_start_str=="")
                                {
                                    orf_start_str=mRNA_start
                                }
                                if(orf_end_str=="")
                                {
                                    orf_end_str=mRNA_start
                                }
                                print($1,mRNA_start,mRNA_end,$4,$5,$6,orf_start_str,orf_end_str,".",exon_index,exon_sizes_str,exon_starts_str)
                            }


                         }' ${exon_file}_merged_temp.bed> ${exon_file}_merged_with_coordinate_file.bed

rm ${exon_file}_merged_temp.bed