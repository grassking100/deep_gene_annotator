bed <- read_bed('~/../home/io/Arabidopsis_thaliana/data/Araport11_coding_gene_2019_04_07.bed')
write_bed(bed[bed$id %in% valid_transcript_id,],'filtered_coding_gene.bed')