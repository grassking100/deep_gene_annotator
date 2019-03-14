args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=1) {
  stop("One arguments must be supplied (input file)", call.=FALSE)
}
saved_root <- args[1]
safe_merged_gro_sites_path <- paste0(saved_root,'safe_merged_gro_sites','.tsv')
safe_merged_cleavage_sites_path <- paste0(saved_root,'safe_merged_cleavage_sites','.tsv')
print('Find genes where TSS and CA sites belong to')
if(file.exists(safe_merged_gro_sites_path) & 
   file.exists(safe_merged_cleavage_sites_path)){
    #print(saved_root)
    print("Result files are already exist,procedure will be skipped.")
} else{
    dist_gro_sites <- belong_by_distance(valid_gro,valid_official_araport11_coding,
                                         -20,-1,'evidence_5_end','exist_5_end',"peak_id",'id')
    #consist(dist_gro_sites,'ref_name','tag_count')
    dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_araport11_coding,
                                              1,20,"evidence_3_end",'exist_3_end',"cleavage_id",'id')
    inner_gro_sites <- belong_by_boundary(valid_gro,valid_external_five_UTR,
                                           'evidence_5_end','start','end','peak_id','id')
    inner_cleavage_sites <- belong_by_boundary(valid_cleavage_site,valid_external_three_UTR,
                                               'evidence_3_end','start','end','cleavage_id','id')
    print("Merge belonging data")
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
    print('Find sites which is far away from gene')
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
    print('Clean and export data')
    long_dist_gro_sites_gene_id <- unique(long_dist_gro_sites$gene_id)
    long_dist_cleavage_sites_gene_id <- unique(long_dist_cleavage_sites$gene_id)
    safe_merged_gro_sites <- merged_gro_sites[!merged_gro_sites$gene_id %in% long_dist_gro_sites_gene_id,]
    safe_merged_cleavage_sites <- merged_cleavage_sites[!merged_cleavage_sites$gene_id %in% long_dist_cleavage_sites_gene_id,]
    write.table(safe_merged_gro_sites,safe_merged_gro_sites_path,sep='\t')
    write.table(safe_merged_cleavage_sites,safe_merged_cleavage_sites_path,sep='\t')
}