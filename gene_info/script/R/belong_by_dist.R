print('Find belong by distance')
dist_gro_sites <- belong_by_distance(valid_gro,valid_official_araport11_coding,
                                     -20,-1,'evidence_5_end','exist_5_end',"peak_id",'id')

dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_araport11_coding,
                                          1,20,"evidence_3_end",'exist_3_end',"cleavage_id",'id')

dist_gro_sites$gro_source <- 'dist'
dist_cleavage_sites$cleavage_source <- 'dist'
write.table(dist_gro_sites,'dist_gro_sites.tsv',sep='\t',quote =F)
write.table(dist_cleavage_sites,'dist_cleavage_sites.tsv',sep='\t',quote =F)

print('Find outer data')
long_dist_gro_sites <- belong_by_distance(valid_gro,valid_official_araport11_coding,
                                          -1000,-21,'evidence_5_end','exist_5_end',
                                          "peak_id",'id')
long_dist_gro_sites$gro_source <- 'long_dist'
#long_dist_gro_sites$gene_id <- find_gene_id(long_dist_gro_sites$ref_name)

long_dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_araport11_coding,
                                               21,500,"evidence_3_end",'exist_3_end',
                                               "cleavage_id",'id')
long_dist_cleavage_sites$cleavage_source <- 'long_dist'
#long_dist_cleavage_sites$gene_id <- find_gene_id(long_dist_cleavage_sites$ref_name)

write.table(long_dist_gro_sites,'long_dist_gro_sites.tsv',sep='\t',quote =F)
write.table(long_dist_cleavage_sites,'long_dist_cleavage_sites.tsv',sep='\t',quote =F)