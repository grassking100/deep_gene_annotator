print("Find belong end")
long_dist_gro_sites['gene_id'] <- NULL
long_dist_cleavage_sites['gene_id'] <- NULL
long_dist_cleavage_sites$cleavage_source <- long_dist_cleavage_sites$gro_source
long_dist_cleavage_sites['gro_source'] <- NULL
merged_gro_sites_ <- rbind(dist_gro_sites,inner_gro_sites,long_dist_gro_sites)
merged_cleavage_sites_ <- rbind(dist_cleavage_sites,inner_cleavage_sites,long_dist_cleavage_sites)

print("Find safe GRO")
merged_gro_sites_ <- merged_gro_sites_[!is.na(merged_gro_sites_['ref_name']),]
merged_gro_sites <- consist(merged_gro_sites_,'ref_name','tag_count')
safe_merged_gro_sites <- merged_gro_sites[merged_gro_sites$gro_source %in% c('inner','dist'),]
print('Find GRO gene id')
safe_merged_gro_sites$gene_id <- find_gene_id(safe_merged_gro_sites$ref_name)
write.table(safe_merged_gro_sites,'safe_merged_gro_sites.tsv',sep='\t',quote =F)

print("Find safe CS")
merged_cleavage_sites_ <- merged_cleavage_sites_[!is.na(merged_cleavage_sites_['ref_name']),]
merged_cleavage_sites <- consist(merged_cleavage_sites_,'ref_name','read_count')
safe_merged_cleavage_sites <- merged_cleavage_sites[merged_cleavage_sites$cleavage_source %in% c('inner','dist'),]
print('Find CS gene id')
safe_merged_cleavage_sites$gene_id <- find_gene_id(safe_merged_cleavage_sites$ref_name)

write.table(safe_merged_cleavage_sites,'safe_merged_cleavage_sites.tsv',sep='\t',quote =F)
