setwd('~/../home')
source('./sequence_annotation/gene_info/script/R/gff.R')
classify_data_by_id <- function(all_id,selected_id,mRNA_bed,coordinate_consist_bed,want_bed_path,unwant_bed_path,id_convert){
    want_id <- selected_id
    unwant_id <- all_id[!all_id %in% selected_id]
    want_bed = coordinate_consist_bed[coordinate_consist_bed$id %in% want_id,]
    unwant_bed = create_unwant_bed(mRNA_bed,coordinate_consist_bed,unwant_id)
    write_bed(want_bed,want_bed_path)
    write_bed(unwant_bed,unwant_bed_path)
}
create_unwant_bed <- function(mRNA_bed,coordinate_consist_bed,unwant_id){
    unwant_bed = rbind(mRNA_bed[mRNA_bed$id %in% unwant_id,],
                       coordinate_consist_bed[coordinate_consist_bed$id %in% unwant_id,])
    return(unwant_bed)
}

args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=7) {
  stop("Seven arguments must be supplied (input file)", call.=FALSE)
}
raw_mRNA_bed_path <- args[1]
coordinate_consist_bed_path <- args[2]
fai_path <- args[3]
upstream_dist <- strtoi(args[4])
downstream_dist <- strtoi(args[5])
result_path <- args[6]
id_convert_path <- args[7]
###Read data###
raw_mRNA_bed <- read_bed12(raw_mRNA_bed_path)
coordinate_consist_bed <- read_bed12(coordinate_consist_bed_path)
want_id <- coordinate_consist_bed[4]
id_convert <- read.table(id_convert_path,stringsAsFactors=F)
#coordinate_consist_with_gene_id <- coordinate_consist_bed
#coordinate_consist_with_gene_id$id <- id_convert[coordinate_consist_bed$id,]
all_id <- raw_mRNA_bed$id
want_bed_path = paste0(result_path,'/want.bed')
unwant_bed_path = paste0(result_path,'/unwant_bed.bed')
saved_num <- -1
index <- 0
###Recurrent cleaning###
while(T){
    index <- index+1
    classify_data_by_id(all_id,want_id,raw_mRNA_bed,coordinate_consist_bed,want_bed_path,unwant_bed_path,id_convert)
    command = paste0("bash ./sequence_annotation/gene_info/script/bash/safe_filter.sh",
                     want_bed_path,unwant_bed_path,fai_path,upstream_dist,downstream_dist)
    system(command)
    id_path = paste0('$region_upstream_',upstream_dist,'_downstream_',downstream_dist,'_safe_zone_id.txt')
    want_id <- read.table(id_path)$V1
    num <- nrow(want_id)
    if (num==saved_num){    
        break
    }
    else{
        saved_num <- num
        want_bed_path = paste0(result_path,'/want_iter_',index,'.bed')
        unwant_bed_path = paste0(result_path,'/unwan_iter_',index,'.bed')
    }
}
###Write data###
want_bed = coordinate_consist_bed[coordinate_consist_bed$id %in% want_id,]
write_bed(want_bed,paste0(result_path,'/recurrent_cleaned.bed'))
#cp ${separate_path}/want_iter_${index}.bed ${result_path}/mRNA_coordinate.bed