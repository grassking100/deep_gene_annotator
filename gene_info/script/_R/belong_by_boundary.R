print('Find belong by boundary')
inner_gro_sites <- belong_by_boundary(valid_gro,valid_external_five_UTR,
                                       'evidence_5_end','start','end','peak_id','id')

inner_cleavage_sites <- belong_by_boundary(valid_cleavage_site,valid_external_three_UTR,
                                           'evidence_3_end','start','end','cleavage_id','id')
inner_cleavage_sites$cleavage_source <- 'inner'
inner_gro_sites$gro_source <- 'inner'
write.table(inner_cleavage_sites,'inner_cleavage_sites.tsv',sep='\t',quote =F)
write.table(inner_gro_sites,'inner_gro_sites.tsv',sep='\t',quote =F)

