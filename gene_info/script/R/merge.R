print('Find belong')
#inner_gro_sites <- belong_by_boundary(valid_gro,valid_external_five_UTR,
#                                       'evidence_5_end','start','end','peak_id','id')
#dist_gro_sites <- belong_by_distance(valid_gro,valid_official_araport11_coding,
#                                     -20,-1,'evidence_5_end','exist_5_end',"peak_id",'id')

#dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_araport11_codi#ng,
#                                          1,20,"evidence_3_end",'exist_3_end',"cleavage_id",'id')

#inner_cleavage_sites <- belong_by_boundary(valid_cleavage_site,valid_external_three_UTR,
#                                           'evidence_3_end','start','end','cleavage_id','id')
print("Find belong end")

#dist_gro_sites$gro_source <- 'dist'
#inner_gro_sites$gro_source <- 'inner'
#merged_gro_sites_ <- rbind(dist_gro_sites,inner_gro_sites)
#merged_gro_sites_ <- merged_gro_sites_[!is.na(merged_gro_sites_['ref_name']),]
#merged_gro_sites <- consist(merged_gro_sites_,'ref_name','tag_count')

#merged_gro_sites$gene_id <- find_gene_id(merged_gro_sites$ref_name)

#dist_cleavage_sites$cleavage_source <- 'dist'
#inner_cleavage_sites$cleavage_source <- 'inner'
#merged_cleavage_sites_ <- rbind(dist_cleavage_sites,inner_cleavage_sites)
#merged_cleavage_sites_ <- merged_cleavage_sites_[!is.na(merged_cleavage_sites_['ref_name']),]
#merged_cleavage_sites <- consist(merged_cleavage_sites_,'ref_name','read_count')
#merged_cleavage_sites$gene_id <- find_gene_id(merged_cleavage_sites$ref_name)
print('Find outer data')
#long_dist_gro_sites <- belong_by_distance(valid_gro,valid_official_araport11_coding,
#                                          -1000,-21,'evidence_5_end','exist_5_end',
#                                          "peak_id",'id')
#long_dist_gro_sites$gro_source <- 'long_dist'
#long_dist_gro_sites$gene_id <- find_gene_id(long_dist_gro_sites$ref_name)

#long_dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_araport11_coding,
#                                               21,500,"evidence_3_end",'exist_3_end',
#                                               "cleavage_id",'id')

#long_dist_cleavage_sites$gro_source <- 'long_dist'
#long_dist_cleavage_sites$gene_id <- find_gene_id(long_dist_cleavage_sites$ref_name)

#write.table(dist_gro_sites,str_with_time('dist_gro_sites_','.tsv'),sep='\t',quote =F)
#write.table(inner_gro_sites,str_with_time('inner_gro_sites_','.tsv'),sep='\t',quote =F)
#write.table(merged_gro_sites,str_with_time('merged_gro_sites_','.tsv'),sep='\t',quote =F)
#write.table(long_dist_gro_sites,str_with_time('long_dist_gro_sites_','.tsv'),sep='\t',quote =F)


#write.table(dist_cleavage_sites,str_with_time('dist_cleavage_sites_','.tsv'),sep='\t',quote =F)
#write.table(inner_cleavage_sites,str_with_time('inner_cleavage_sites_','.tsv'),sep='\t',quote =F)
#write.table(merged_cleavage_sites,str_with_time('merged_cleavage_sites_','.tsv'),sep='\t',quote =F)

print("Write")
#write.table(long_dist_cleavage_sites,str_with_time('long_dist_cleavage_sites_','.tsv'),sep='\t',quote =F)
print("Write end")

long_dist_gro_sites <- read.csv('script/data/2019_04_03/long_dist_gro_sites_2019_04_03.tsv',sep='\t',stringsAsFactors=F)
long_dist_cleavage_sites <- read.csv('script/data/2019_04_03/long_dist_cleavage_sites_2019_04_03.tsv',sep='\t',stringsAsFactors=F)
merged_gro_sites <- read.csv('script/data/2019_04_03/merged_gro_sites_2019_04_03.tsv',sep='\t',stringsAsFactors=F)
merged_cleavage_sites <- read.csv('script/data/2019_04_03/merged_cleavage_sites_2019_04_03.tsv',sep='\t',stringsAsFactors=F)

############################################################################################
print('Clean and merge data')
long_dist_gro_sites_gene_id <- unique(long_dist_gro_sites$gene_id)
long_dist_cleavage_sites_gene_id <- unique(long_dist_cleavage_sites$gene_id)
safe_merged_gro_sites <- merged_gro_sites[!merged_gro_sites$gene_id %in% long_dist_gro_sites_gene_id,]
safe_merged_cleavage_sites <- merged_cleavage_sites[!merged_cleavage_sites$gene_id %in% long_dist_cleavage_sites_gene_id,]

write.table(safe_merged_gro_sites,str_with_time('script/data/2019_04_03/safe_merged_gro_sites_','.tsv'),sep='\t',quote =F)
write.table(safe_merged_cleavage_sites,str_with_time('script/data/2019_04_03/safe_merged_cleavage_sites_','.tsv'),sep='\t',quote =F)

merged_data <- merge(safe_merged_gro_sites,safe_merged_cleavage_sites,
                     c('chr','strand','ref_name','gene_id'),all=T)
write.table(merged_data,str_with_time('non_unique_merged_data_','.tsv'),sep='\t',quote =F)
merged_data <- unique(merged_data)
write.table(merged_data,str_with_time('merged_data_','.tsv'),sep='\t',quote =F)

clean_merged_data <- merged_data[!is.na(merged_data$evidence_5_end) & !is.na(merged_data$evidence_3_end),]
clean_merged_data$coordinate_start <- apply(clean_merged_data[,c('evidence_5_end','evidence_3_end')],1,min)
clean_merged_data$coordinate_end <- apply(clean_merged_data[,c('evidence_5_end','evidence_3_end')],1,max)
write.table(clean_merged_data,str_with_time('script/data/2019_04_03/clean_merged_data_','.tsv'),sep='\t',quote =F)
############################################################################################
print('Consist with gene')

clean_merged_data2 <- consist(clean_merged_data,'gene_id','tag_count',remove_duplicate = F)
print(1)
clean_merged_data3 <- consist(clean_merged_data2,'gene_id','read_count',remove_duplicate = F)
print(2)
consist_data_ <- coordinate_consist_filter(clean_merged_data3,'gene_id','coordinate_start')
print(3)
consist_data <- coordinate_consist_filter(consist_data_,'gene_id','coordinate_end')
print(4)

write.table(clean_merged_data2,str_with_time('clean_merged_data2_','.tsv'),sep='\t',quote =F)
write.table(clean_merged_data3,str_with_time('clean_merged_data3_','.tsv'),sep='\t',quote =F)
write.table(consist_data_,str_with_time('consist_data_','.tsv'),sep='\t',quote =F)
write.table(consist_data,str_with_time('consist_data2_','.tsv'),sep='\t',quote =F)


write.table(safe_merged_gro_sites,str_with_time('safe_merged_gro_sites_','.tsv'),sep='\t')
write.table(safe_merged_cleavage_sites,str_with_time('safe_merged_cleavage_sites_','.tsv'),sep='\t')