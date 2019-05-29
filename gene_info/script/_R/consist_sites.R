setwd('~/../home')
source('./sequence_annotation/gene_info/script/R/belong_finder.R')
source('./sequence_annotation/gene_info/script/R/merge_and_clean.R')
args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=7) {
  stop("Seven arguments must be supplied (input file)", call.=FALSE)
}
dist_gro_sites_path <- args[1]
dist_cleavage_sites_path <- args[2]
inner_gro_sites_path <- args[3]
inner_cleavage_sites_path <- args[4]
long_dist_gro_sites_path <- args[5]
long_dist_cleavage_sites_path <- args[6]
saved_root <- args[7]

safe_merged_gro_sites_path <- paste0(saved_root,'/safe_merged_gro_sites','.tsv')
safe_merged_cleavage_sites_path <- paste0(saved_root,'/safe_merged_cleavage_sites','.tsv')
print('Find TSS and CA sites are belong to which genes')
if(file.exists(safe_merged_gro_sites_path) & 
   file.exists(safe_merged_cleavage_sites_path)){
    print("Result files are already exist, procedure will be skipped.")
} else{
    ###Read file###
    dist_gro_sites = read.table(dist_gro_sites_path,sep='\t',header=T)
    dist_cleavage_sites = read.table(dist_cleavage_sites_path,sep='\t',header=T)
    inner_gro_sites = read.table(inner_gro_sites_path,sep='\t',header=T)
    inner_cleavage_sites = read.table(inner_cleavage_sites_path,sep='\t',header=T)
    long_dist_gro_sites = read.table(long_dist_gro_sites_path,sep='\t',header=T)
    long_dist_cleavage_sites = read.table(long_dist_cleavage_sites_path,sep='\t',header=T)
    
    #Assign valid GRO sites and cleavage sites to gene###
    print("Merge belonging data")
    dist_gro_sites$gro_source <- 'dist'
    inner_gro_sites$gro_source <- 'inner'
    merged_gro_sites_ <- rbind(dist_gro_sites,inner_gro_sites)
    merged_gro_sites_ <- merged_gro_sites_[!is.na(merged_gro_sites_['ref_name']),]
    merged_gro_sites <- consist(merged_gro_sites_,'ref_name','tag_count')
    merged_gro_sites$gene_id <- find_gene_id(merged_gro_sites$ref_name)
    dist_cleavage_sites$cleavage_source <- 'dist'
    inner_cleavage_sites$cleavage_source <- 'inner'
    merged_cleavage_sites_ <- rbind(dist_cleavage_sites,inner_cleavage_sites)
    merged_cleavage_sites_ <- merged_cleavage_sites_[!is.na(merged_cleavage_sites_['ref_name']),]
    merged_cleavage_sites <- consist(merged_cleavage_sites_,'ref_name','read_count')
    merged_cleavage_sites$gene_id <- find_gene_id(merged_cleavage_sites$ref_name)

    ###Clean data without invalid sites###
    print('Clean and export data')
    long_dist_gro_sites_gene_id <- unique(long_dist_gro_sites$gene_id)
    long_dist_cleavage_sites_gene_id <- unique(long_dist_cleavage_sites$gene_id)
    safe_merged_gro_sites <- merged_gro_sites[!merged_gro_sites$gene_id %in% long_dist_gro_sites_gene_id,]
    safe_merged_cleavage_sites <- merged_cleavage_sites[!merged_cleavage_sites$gene_id %in% long_dist_cleavage_sites_gene_id,]
    ###Write data###
    write.table(safe_merged_gro_sites,safe_merged_gro_sites_path,sep='\t')
    write.table(safe_merged_cleavage_sites,safe_merged_cleavage_sites_path,sep='\t')
}