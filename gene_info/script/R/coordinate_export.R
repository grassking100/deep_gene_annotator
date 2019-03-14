coordinate_export <- function(safe_merged_gro_sites,safe_merged_cleavage_sites,saved_root)
{
    coordinate_consist_bed_path <- paste0(saved_root,
                                          str_with_time('coordinate_consist_with_','.bed'))
    coordinate_consist_bed_with_gene_id_path <- paste0(saved_root,
                                                       str_with_time('coordinate_consist_with_gene_id_',
                                                                     '.bed'))
    pure_coordinate_consist_bed_path <- paste0(saved_root,
                                               str_with_time('pure_coordinate_consist_','.bed'))
    merged_data <- merge(safe_merged_gro_sites,safe_merged_cleavage_sites,
                         c('chr','strand','ref_name','gene_id'),all=T)

    merged_data <- unique(merged_data)
    clean_merged_data <- merged_data[!is.na(merged_data$evidence_5_end) & !is.na(merged_data$evidence_3_end),]
    clean_merged_data$coordinate_start <- apply(clean_merged_data[,c('evidence_5_end',
                                                                     'evidence_3_end')],1,min)
    clean_merged_data$coordinate_end <- apply(clean_merged_data[,c('evidence_5_end',
                                                                   'evidence_3_end')],1,max)
    print('Consist data with gene id')
    clean_merged_data_by_id <- consist(clean_merged_data,'gene_id','tag_count',remove_duplicate = F)
    clean_merged_data_by_id <- consist(clean_merged_data_by_id,'gene_id','read_count',remove_duplicate = F)
    consist_data <- coordinate_consist_filter(clean_merged_data_by_id,'gene_id','coordinate_start')
    consist_data <- coordinate_consist_filter(clean_merged_data_by_id,'gene_id','coordinate_end')
    
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
    write_bed(coordinate_consist_bed,coordinate_consist_bed_path)

    coordinate_consist_bed_with_gene_id = gff_to_bed(coordinate_consist_gff,'group')
    coordinate_consist_bed_with_gene_id$group = find_gene_id(coordinate_consist_bed_with_gene_id$group)
    coordinate_consist_bed_with_gene_id <- unique(coordinate_consist_bed_with_gene_id)
    write_bed(coordinate_consist_bed_with_gene_id,coordinate_consist_bed_with_gene_id_path)

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
    write_bed(pure_coordinate_consist_bed,pure_coordinate_consist_bed_path)
}

args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=1) {
  stop("One arguments must be supplied (input file)", call.=FALSE)
}
saved_root <- args[1]
coordinate_consist_bed_path <- paste0(saved_root,'coordinate_consist','.bed')
coordinate_consist_bed_with_gene_id_path <- paste0(saved_root,'coordinate_consist_with_gene_id','.bed')
pure_coordinate_consist_bed_path <- paste0(saved_root,'pure_coordinate_consist','.bed')
print('Use TSS and CA sites to create gene coordinate data and export it')
if(file.exists(coordinate_consist_bed_path) & 
   file.exists(coordinate_consist_bed_with_gene_id_path) &
   file.exists(pure_coordinate_consist_bed_path)
  ){
    print("Result files are already exist,procedure will be skipped.")
}else{
    
    safe_merged_gro_sites_path <- paste0(saved_root,str_with_time('safe_merged_gro_sites_','.tsv'))
    safe_merged_cleavage_sites_path <- paste0(saved_root,
                                              str_with_time('safe_merged_cleavage_sites_','.tsv'))  
    safe_merged_gro_sites <- read.table(safe_merged_gro_sites_path,sep='\t')
    safe_merged_cleavage_sites <- read.table(safe_merged_cleavage_sites_path,sep='\t')
    coordinate_export(safe_merged_gro_sites,safe_merged_cleavage_sites,saved_root)
}

          