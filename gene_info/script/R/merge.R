print('Find belong')

dist_gro_sites <- belong_by_distance(valid_gro,valid_official_araport11_coding,
                                     -20,-1,'evidence_5_end','exist_5_end',"peak_id",'id')
#consist(dist_gro_sites,'ref_name','tag_count')
dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_araport11_coding,
                                          1,20,"evidence_3_end",'exist_3_end',"cleavage_id",'id')
inner_gro_sites <- belong_by_boundary(valid_gro,valid_external_five_UTR,
                                       'evidence_5_end','start','end','peak_id','id')
inner_cleavage_sites <- belong_by_boundary(valid_cleavage_site,valid_external_three_UTR,
                                           'evidence_3_end','start','end','cleavage_id','id')
print("Find belong end")

dist_gro_sites$gro_source <- 'dist'
inner_gro_sites$gro_source <- 'inner'
merged_gro_sites_ <- rbind(dist_gro_sites,inner_gro_sites)
merged_gro_sites_ <- merged_gro_sites_[!is.na(merged_gro_sites_['ref_name']),]
merged_gro_sites <- consist(merged_gro_sites_,'ref_name','tag_count')
#print(merged_gro_sites)
merged_gro_sites$gene_id <- find_gene_id(merged_gro_sites$ref_name)

dist_cleavage_sites$cleavage_source <- 'dist'
inner_cleavage_sites$cleavage_source <- 'inner'
merged_cleavage_sites_ <- rbind(dist_cleavage_sites,inner_cleavage_sites)
merged_cleavage_sites_ <- merged_cleavage_sites_[!is.na(merged_cleavage_sites_['ref_name']),]
merged_cleavage_sites <- consist(merged_cleavage_sites_,'ref_name','read_count')
merged_cleavage_sites$gene_id <- find_gene_id(merged_cleavage_sites$ref_name)
print('Find outer data')
long_dist_gro_sites <- belong_by_distance(valid_gro,valid_official_araport11_coding,
                                          -1000,-21,'evidence_5_end','exist_5_end',
                                          "peak_id",'id')
long_dist_gro_sites$gro_source <- 'long_dist'
long_dist_gro_sites$gene_id <- find_gene_id(long_dist_gro_sites$ref_name)

long_dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_araport11_coding,
                                               21,500,"evidence_3_end",'exist_3_end',
                                               "cleavage_id",'id')

long_dist_cleavage_sites$gro_source <- 'long_dist'
long_dist_cleavage_sites$gene_id <- find_gene_id(long_dist_cleavage_sites$ref_name)

############################################################################################
print('Clean and merge data')
long_dist_gro_sites_gene_id <- unique(long_dist_gro_sites$gene_id)
long_dist_cleavage_sites_gene_id <- unique(long_dist_cleavage_sites$gene_id)
safe_merged_gro_sites <- merged_gro_sites[!merged_gro_sites$gene_id %in% long_dist_gro_sites_gene_id,]
safe_merged_cleavage_sites <- merged_cleavage_sites[!merged_cleavage_sites$gene_id %in% long_dist_cleavage_sites_gene_id,]

merged_data <- merge(safe_merged_gro_sites,safe_merged_cleavage_sites,
                     c('chr','strand','ref_name','gene_id'),all=T)

merged_data <- unique(merged_data)
clean_merged_data <- merged_data[!is.na(merged_data$evidence_5_end) & !is.na(merged_data$evidence_3_end),]
clean_merged_data$coordinate_start <- apply(clean_merged_data[,c('evidence_5_end','evidence_3_end')],1,min)
clean_merged_data$coordinate_end <- apply(clean_merged_data[,c('evidence_5_end','evidence_3_end')],1,max)
############################################################################################
print('Consist with gene')
clean_merged_data_by_id <- consist(clean_merged_data,'gene_id','tag_count',remove_duplicate = F)
clean_merged_data_by_id <- consist(clean_merged_data_by_id,'gene_id','read_count',remove_duplicate = F)
consist_data <- coordinate_consist_filter(clean_merged_data_by_id,'gene_id','coordinate_start')
consist_data <- coordinate_consist_filter(clean_merged_data_by_id,'gene_id','coordinate_end')
write.table(safe_merged_gro_sites,str_with_time('safe_merged_gro_sites_','.tsv'),sep='\t')
write.table(safe_merged_cleavage_sites,str_with_time('safe_merged_cleavage_sites_','.tsv'),sep='\t')