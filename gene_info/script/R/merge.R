print("Merge data")
merged_data <- merge(safe_merged_gro_sites,safe_merged_cleavage_sites,
                     c('chr','strand','ref_name','gene_id'),all=T)
write.table(merged_data,'non_unique_merged_data.tsv',sep='\t',quote =F)
merged_data <- unique(merged_data)
write.table(merged_data,'merged_data.tsv',sep='\t',quote =F)

clean_merged_data <- merged_data[!is.na(merged_data$evidence_5_end) & !is.na(merged_data$evidence_3_end),]
clean_merged_data$coordinate_start <- apply(clean_merged_data[,c('evidence_5_end','evidence_3_end')],1,min)
clean_merged_data$coordinate_end <- apply(clean_merged_data[,c('evidence_5_end','evidence_3_end')],1,max)
write.table(clean_merged_data,'clean_merged_data.tsv',sep='\t',quote =F)
