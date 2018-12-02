args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=3) {
  stop("Three argument must be supplied (input file)", call.=FALSE)
}
ids <- args[1]
exon_bed <- args[2]
selected_exon_path <- args[3]
exon_bed <- read_bed(exon_bed)
print(head(exon_bed))
selected_exon <- exon_bed[exon_bed[,'id'] %in% ids,]
write_bed(selected_exon,selected_exon_path)