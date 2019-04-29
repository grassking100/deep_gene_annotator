args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=3) {
  stop("Nine arguments must be supplied (input file)", call.=FALSE)
}
raw_mRNA_bed_path <- args[1]
coordinate_consist_bed_path <- args[2]
fixed_bed_path <- args[3]

raw_mRNA_bed <- read_bed(raw_mRNA_bed_path)
coordinate_consist_bed <- read_bed(coordinate_consist_bed_path)
left = raw_mRNA_bed[!raw_mRNA_bed$id %in% coordinate_consist_bed$id,]
fixed_bed <- rbind(left,coordinate_consist_bed)
write_bed(fixed_bed,fixed_bed_path)
