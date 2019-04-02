setwd('~/../home')
source('./sequence_annotation/gene_info/script/R/belong_finder.R')
args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=9) {
  stop("Nine arguments must be supplied (input file)", call.=FALSE)
}
valid_official_gene_info_path <- args[1]
valid_gro_site_path <- args[2]
valid_cleavage_site_path <- args[3]
valid_external_five_UTR_path <- args[4]
valid_external_three_UTR_path <- args[5]
saved_root <- args[6]
upstream_dist <- strtoi(args[7])
downstream_dist <- strtoi(args[8])
tolerate_dist <- strtoi(args[9])

dist_gro_sites_path <- paste0(saved_root,'/dist_gro_sites','.tsv')
dist_cleavage_sites_path <- paste0(saved_root,'/dist_cleavage_sites','.tsv')
inner_gro_sites_path <- paste0(saved_root,'/inner_gro_sites','.tsv')
inner_cleavage_sites_path <- paste0(saved_root,'/inner_cleavage_sites','.tsv')
long_dist_gro_sites_path <- paste0(saved_root,'/long_dist_gro_sites','.tsv')
long_dist_cleavage_sites_path <- paste0(saved_root,'/long_dist_cleavage_sites','.tsv')
print('Find TSS and CA sites are belong to which genes')
if(file.exists(dist_gro_sites_path) &
   file.exists(dist_cleavage_sites_path) &
   file.exists(inner_gro_sites_path) &
   file.exists(inner_cleavage_sites_path) &
   file.exists(long_dist_gro_sites_path) &
   file.exists(long_dist_cleavage_sites_path)
  ){
    print("Result files are already exist, procedure will be skipped.")
} else{
    ###Read file###
    valid_official_gene_info = read.table(valid_official_gene_info_path,sep='\t',header=T)
    valid_gro = read.table(valid_gro_site_path,sep='\t',header=T)
    valid_cleavage_site = read.table(valid_cleavage_site_path,sep='\t',header=T)
    valid_external_five_UTR = read.table(valid_external_five_UTR_path,sep='\t',header=T)
    valid_external_three_UTR = read.table(valid_external_three_UTR_path,sep='\t',header=T)
    print('Classify valid GRO sites and cleavage sites and write data')
    ###Classify valid GRO sites and cleavage sites and write data###
    if(!file.exists(dist_gro_sites_path)){
        dist_gro_sites <- belong_by_distance(valid_gro,valid_official_gene_info,
                                             -tolerate_dist,-1,'evidence_5_end','exist_5_end',"peak_id",'id')
        write.table(dist_gro_sites,dist_gro_sites_path,quote=F,row.names=F,sep='\t')
    }
    if(!file.exists(dist_cleavage_sites_path)){
    dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_gene_info,
                                              1,tolerate_dist,"evidence_3_end",'exist_3_end',"cleavage_id",'id')
     write.table(dist_cleavage_sites,dist_cleavage_sites_path,quote=F,row.names=F,sep='\t')
    }
    if(!file.exists(inner_gro_sites_path)){
    inner_gro_sites <- belong_by_boundary(valid_gro,valid_external_five_UTR,
                                           'evidence_5_end','start','end','peak_id','id')
        write.table(inner_gro_sites,inner_gro_sites_path,quote=F,row.names=F,sep='\t')
    }
    if(!file.exists(inner_cleavage_sites_path)){
    inner_cleavage_sites <- belong_by_boundary(valid_cleavage_site,valid_external_three_UTR,
                                               'evidence_3_end','start','end','cleavage_id','id')
    write.table(inner_cleavage_sites,inner_cleavage_sites_path,quote=F,row.names=F,sep='\t')
    }
    ###Search invalid GRO sites and cleavage sites###
    print('Find sites which are far away from gene')
     if(!file.exists(long_dist_gro_sites_path)){
    long_dist_gro_sites <- belong_by_distance(valid_gro,valid_official_gene_info,
                                              -upstream_dist,-tolerate_dist-1,'evidence_5_end','exist_5_end',
                                              "peak_id",'id')
    long_dist_gro_sites$gro_source <- 'long_dist'
    #long_dist_gro_sites$gene_id <- find_gene_id(long_dist_gro_sites$ref_name)
         write.table(long_dist_gro_sites,long_dist_gro_sites_path,quote=F,row.names=F,sep='\t')
    }
     if(!file.exists(long_dist_cleavage_sites_path)){
    long_dist_cleavage_sites <- belong_by_distance(valid_cleavage_site,valid_official_gene_info,
                                                   tolerate_dist+1,downstream_dist,"evidence_3_end",'exist_3_end',
                                                   "cleavage_id",'id')
    long_dist_cleavage_sites$gro_source <- 'long_dist'
    #long_dist_cleavage_sites$gene_id <- find_gene_id(long_dist_cleavage_sites$ref_name)
         write.table(long_dist_cleavage_sites,long_dist_cleavage_sites_path,quote=F,row.names=F,sep='\t')
     }
}