setwd('~/../home')
source('./sequence_annotation/gene_info/script/R/gff.R')
args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=1) {
  stop("One argument must be supplied (input file)", call.=FALSE)
}

bed <- read_bed(args[1])
temp_bed <- bed[,c(1,2,3,6)]
unique_bed <- bed[!duplicated(temp_bed),]
path_root <- gsub(".bed", "", args[1])
write_bed(unique_bed,paste0(path_root,'_unique_simple_coordinate.bed'))