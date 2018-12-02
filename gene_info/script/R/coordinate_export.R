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
coordinate_consist_bed <- unique(coordinate_consist_bed)
write_bed(coordinate_consist_bed,str_with_time('coordinate_consist_with_','.bed'))

coordinate_consist_bed_with_gene_id = gff_to_bed(coordinate_consist_gff,'group')
coordinate_consist_bed_with_gene_id$group = find_gene_id(coordinate_consist_bed_with_gene_id$group)
coordinate_consist_bed_with_gene_id <- unique(coordinate_consist_bed_with_gene_id)
write_bed(coordinate_consist_bed_with_gene_id,str_with_time('coordinate_consist_with_gene_id_','.bed'))

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
write_bed(pure_coordinate_consist_bed,str_with_time('pure_coordinate_consist_','.bed'))

          

          