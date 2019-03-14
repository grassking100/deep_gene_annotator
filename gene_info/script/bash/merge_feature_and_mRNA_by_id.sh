feature_file="${1%.*}"
coordinate_file="${2%.*}"
saved_root=$3
saved_file=$4
feature=${5:-exon}

awk -F'\t' -v OFS="\t" '{gsub("\r", "",$0); print $1,$2,$3,$4,$5,$6,".",".","feature"}' ${feature_file}.bed > ${feature_file}.bed_extend

awk -F'\t' -v OFS="\t" '{gsub("\r", "",$0); print $1,$2,$3,$4,$5,$6,$7,$8,"mRNA"}' ${coordinate_file}.bed > ${coordinate_file}.bed_extend

cat ${feature_file}.bed_extend ${coordinate_file}.bed_extend | sort -k4,4 | bedtools groupby -g 4 -c 1,2,3,5,6,7,8,9 -o distinct,collapse,collapse,distinct,distinct,distinct,distinct,collapse | awk -F'\t' -v OFS="\t"  '{print $2,$3,$4,$1,".",$6,$7,$8,$9}' > ${feature_file}_merged_temp.bed

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
                                target_index=0
                                for(i=1;i<=n;i++)
                                {
                                    if(types[i]=="feature")
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
                                        target_index += 1
                                        target_starts[target_index] = start
                                        target_ends[target_index] = end

                                    }
                                }
                                if(target_index>0)
                                {
                                    target_start_min = target_starts[1]
                                    target_end_max = target_ends[1]
                                    min_index=1
                                    max_index=1
                                    for(i=2;i<=target_index;i++)
                                    {
                                        if(target_start_min>target_starts[i])
                                        {
                                            target_start_min=target_starts[i]
                                            min_index=i
                                        }
                                        if(target_end_max<target_ends[i])
                                        {
                                            target_end_max=target_ends[i]
                                            max_index=i
                                        }
                                    }
                                    if(target_end_max<mRNA_end)
                                    {
                                        target_ends[max_index]=mRNA_end
                                    }
                                    if(target_start_min>mRNA_start)
                                    {
                                        target_starts[min_index]=mRNA_start
                                    }
                                    target_starts_str = target_starts[1] - mRNA_start
                                    target_sizes_str = target_ends[1] - target_starts[1]
                                    for(i=2;i<=target_index;i++)
                                    {
                                        target_starts_str = target_starts_str "," target_starts[i] - mRNA_start
                                        target_sizes_str = target_sizes_str "," target_ends[i]-target_starts[i]
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
                                    print($1,mRNA_start,mRNA_end,$4,$5,$6,orf_start_str,orf_end_str,".",target_index,target_sizes_str,target_starts_str)
                                }
                            }


                         }' ${feature_file}_merged_temp.bed> ${saved_root}/$saved_file
rm ${feature_file}_merged_temp.bed