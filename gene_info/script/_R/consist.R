############################################################################################
print('Consist with gene')

consist_data_ <- consist(clean_merged_data,'gene_id','tag_count',remove_duplicate = F)
consist_data_ <- consist(consist_data_,'gene_id','read_count',remove_duplicate = F)
consist_data_ <- coordinate_consist_filter(consist_data_,'gene_id','coordinate_start')
consist_data <- coordinate_consist_filter(consist_data_,'gene_id','coordinate_end')

write.table(consist_data,'consist_data.tsv',sep='\t',quote =F)

coordinate_consist_gff <- data.frame(chr=paste('Chr',consist_data$chr,sep=''),
                                     source='Reannotated Araport11',
                                          feature='transcription',
                                          start=consist_data$coordinate_start,
                                          end=consist_data$coordinate_end,
                                          score='.',
                                          strand=consist_data$strand,
                                          frame='.',
                                          group=consist_data$ref_name,stringsAsFactors = F
)


coordinate_consist_bed = gff_to_bed(coordinate_consist_gff,'group')
coordinate_consist_bed$chr <- apply(coordinate_consist_bed,
                                    1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
coordinate_consist_bed <- unique(coordinate_consist_bed)
write_bed(coordinate_consist_bed,'coordinate_consist.bed')

coordinate_consist_bed_with_gene_id = gff_to_bed(coordinate_consist_gff,'group')
coordinate_consist_bed_with_gene_id$chr <- apply(coordinate_consist_bed_with_gene_id,
                                                 1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
coordinate_consist_bed_with_gene_id$group = find_gene_id(coordinate_consist_bed_with_gene_id$group)
coordinate_consist_bed_with_gene_id <- unique(coordinate_consist_bed_with_gene_id)
write_bed(coordinate_consist_bed_with_gene_id,'coordinate_consist_with_gene_id.bed')

pure_coordinate_consist_gff <- data.frame(chr=paste('Chr',consist_data$chr,sep=''),
                                          source='Reannotated Araport11',
                                          feature='transcription',
                                          start=consist_data$coordinate_start,
                                          end=consist_data$coordinate_end,
                                          score='.',
                                          strand=consist_data$strand,
                                          frame='.',
                                          group=".",stringsAsFactors = F
)
pure_coordinate_consist_bed <- unique(gff_to_bed(pure_coordinate_consist_gff,"group"))
pure_coordinate_consist_bed$chr <- apply(pure_coordinate_consist_bed,
                                         1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
write_bed(pure_coordinate_consist_bed,'pure_coordinate_consist.bed')