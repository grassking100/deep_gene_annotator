args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) {
  stop("Two argument must be supplied (input file).n", call.=FALSE)
}
ids <- args[1]
export_bed_path <- args[2]
ids <- read.table(ids)$V1
bed = coordinate_consist_bed[find_gene_id(coordinate_consist_bed$group) %in% ids,]
write_bed(gff_to_bed(bed,'group'),export_bed_path)