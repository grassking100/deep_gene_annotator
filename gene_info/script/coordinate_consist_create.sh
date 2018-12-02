#Read library
Rscript --save script/R/utils.R
Rscript --save --restore script/R/gff.R
Rscript --save --restore script/R/belong_finder.R
Rscript --save --restore script/R/merge_and_clean.R
#Read and run
Rscript --save --restore script/R/read.R
Rscript --save --restore script/R/merge.R
#Write to coordinate_consist_with_gene_id_${date_}.bed
Rscript --save --restore script/R/coordinate_export.R