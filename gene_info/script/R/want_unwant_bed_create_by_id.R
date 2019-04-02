#!!!Bed's start is zero-based, bed's end is one-based,and gff is one-based coordinated system
#Get want id and unwant id
args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=6) {
  stop("Six arguments must be supplied (input file)", call.=FALSE)
}
want_id <- args[1]
coordinate_consist_bed <- args[2]
mRNA_bed_path <- args[3]
export_want_bed_path <- args[4]
export_unwant_bed_path <- args[5]
convert_want_id_to_gene_id <- as.logical(args[6])
mRNA_bed <- read_bed(mRNA_bed_path)
coordinate_consist_bed <- read_bed(coordinate_consist_bed)
want_id <- read.table(want_id)$V1
if(convert_want_id_to_gene_id)
{
    #
    #want_id <- unique(find_gene_id(want_id))
    want_id <- unique(want_id)
}

###########################################################################################################3
unwant_id <- mRNA_bed[!mRNA_bed$id %in% want_id,]$id
#################################################################
print("Get nonoverlap mRNA and unwant mRNA")
#want_bed = coordinate_consist_bed[find_gene_id(coordinate_consist_bed$id) %in% want_id,]
want_bed = coordinate_consist_bed[coordinate_consist_bed$id %in% want_id,]
#unwant_bed = mRNA_bed[find_gene_id(mRNA_bed$id) %in% unwant_id,]
unwant_bed = mRNA_bed[mRNA_bed$id %in% unwant_id,]
#Export file
print("Export file")
print(export_want_bed_path)
print(export_unwant_bed_path)
want_bed <- gff_to_bed(want_bed,'id',orf_start_convert,orf_end_convert)
want_bed$group = find_gene_id(want_bed$id)
write_bed(unique(want_bed[,1:6]),export_want_bed_path)
write_bed(unwant_bed,export_unwant_bed_path)
print("Export finished")
