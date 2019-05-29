library(stringr)
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2) {
  stop("Two argument must be supplied (input file).n", call.=FALSE)
}

unmerged_bed <- args[1]
merged_bed_path <- args[2]
print(unmerged_bed)
print(merged_bed_path)
merged <- read_bed12(unmerged_bed)
remerged <- reorder_merge_bed(merged)
status <- duplicated(remerged[,c('chr','start','end','score','strand','orf_start','orf_end',
                                 'rgb','count','block_size','block_related_start')])
remerged <- remerged[!status,]
write_bed12(remerged,merged_bed_path)
