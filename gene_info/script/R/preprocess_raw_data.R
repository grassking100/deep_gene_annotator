setwd('~/../home')
source('./sequence_annotation/gene_info/script/R/gff.R')
source('./sequence_annotation/gene_info/script/R/merge_and_clean.R')
args <- commandArgs(trailingOnly=TRUE)
print(args)
if (length(args)!=8) {
  stop("Eight arguments must be supplied (input file)", call.=FALSE)
}
saved_root <- args[1]
bed_path <- args[2]
biomart_path <- args[3]
gro_1_path <- args[4]
gro_2_path <- args[5]
cs_path <- args[6]
most_five_UTR <- args[7]
most_three_UTR <- args[8]

valid_official_araport11_coding_path <- paste0(saved_root,'/valid_official_araport11_coding.tsv')
valid_official_araport11_coding_bed_path <- paste0(saved_root,'/valid_official_araport11_coding.bed')
valid_gro_path <- paste0(saved_root,'/valid_gro.tsv')
valid_cleavage_site_path <- paste0(saved_root,'/valid_cleavage_site.tsv')
valid_external_five_UTR_path <- paste0(saved_root,'/valid_external_five_UTR.tsv')
valid_external_three_UTR_path <- paste0(saved_root,'/valid_external_three_UTR.tsv')
id_convert_path <- paste0(saved_root,'/id_convert.tsv')
if(file.exists(valid_official_araport11_coding_path) & 
   file.exists(valid_official_araport11_coding_bed_path) & 
   file.exists(valid_gro_path) &
   file.exists(valid_cleavage_site_path) &
   file.exists(valid_external_five_UTR_path) &
   file.exists(valid_external_three_UTR_path) &
   file.exists(id_convert_path)
  ){
    print("Result files are already exist, procedure will be skipped.")
} else{
    ###Read file###
    biomart_araport_11_gene_info <- read.csv(biomart_path,stringsAsFactors=F)
    official_araport11_coding <-read_bed12(bed_path)
    gro_1 <- read.csv(gro_1_path,
                      sep='\t',header=T,stringsAsFactors=F,comment.char = "#")
    gro_2 <- read.csv(gro_2_path,
                      sep='\t',header=T,stringsAsFactors=F,comment.char = "#")
    cleavage_site <- read.csv(cs_path,stringsAsFactors=F)
    external_five_UTR <- read.csv(most_five_UTR,sep='\t',header=T,stringsAsFactors=F)
    external_three_UTR <- read.csv(most_three_UTR,sep='\t',header=T,stringsAsFactors=F)
    ###Process araport_11_gene_info###
    biomart_araport_11_gene_info = biomart_araport_11_gene_info[c('Gene.stable.ID','Gene.start..bp.',
                                                                  'Gene.end..bp.','Transcript.stable.ID',
                                                                  'Transcript.start..bp.','Transcript.end..bp.',
                                                                  'Strand','Chromosome.scaffold.name','Transcript.type')]
    names(biomart_araport_11_gene_info) <- c('gene_id','biomart_gene_start','biomart_gene_end',
                                             'transcript_id','biomart_transcript_start',
                                             'biomart_transcript_end','strand','chr','transcript_type')

    biomart_araport_11_gene_info <- subset(biomart_araport_11_gene_info,chr %in% as.character(1:5))
    biomart_araport11_coding <- subset(biomart_araport_11_gene_info,transcript_type == 'protein_coding')
    official_araport11_coding$chr <- apply(official_araport11_coding,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
    official_araport11_coding <- subset(official_araport11_coding,chr %in% as.character(1:5))
    ###Create id_convert table###
    id_convert <- as.character(biomart_araport11_coding$gene_id)
    names(id_convert) <- as.character(biomart_araport11_coding$transcript_id)
    ###Creatre valid geen and mRNA id###
    valid_transcript_id <- intersect(official_araport11_coding$id,biomart_araport11_coding$transcript_id)
    valid_gene_id <- id_convert[as.character(valid_transcript_id)]
    ###Create valid_official_araport11_coding###
    valid_official_araport11_coding <- subset(official_araport11_coding,id %in% valid_transcript_id)
    valid_official_araport11_coding$gene_id <- id_convert[valid_official_araport11_coding$id]
    valid_official_araport11_coding$exist_5_end <- get_five_end(valid_official_araport11_coding,'start','end')
    valid_official_araport11_coding$exist_3_end <- get_three_end(valid_official_araport11_coding,'start','end')
    ###Process GRO sites data###
    gro_ <- rbind(gro_1,gro_2)
    temp <- gro_[c('chr','start','end','strand')]
    gro <- gro_[duplicated(temp),]
    gro <- gro[c('chr','strand','Normalized.Tag.Count','start','end')]
    names(gro) <- c('chr','strand','tag_count','start','end')
    gro['mode'] <- round((gro$end + gro$start)/2)
    valid_gro <- subset(gro,chr %in% as.character(1:5))
    valid_gro <- unique(valid_gro)
    valid_gro$evidence_5_end <- valid_gro$mode
    ###Process cleavage sites data###
    cleavage_site[cleavage_site$Strand=='fwd',]['Strand'] <- '+'
    cleavage_site[cleavage_site$Strand=='rev',]['Strand'] <- '-'
    cleavage_site <- cleavage_site[,c('Chromosome','Strand','Position','Raw.DRS.read.count')]
    colnames(cleavage_site) <- c('chr','strand','position','read_count')
    cleavage_site$chr <- apply(cleavage_site,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
    valid_cleavage_site <- subset(cleavage_site,chr %in% as.character(1:5))
    valid_cleavage_site <- unique(valid_cleavage_site)
    valid_cleavage_site$evidence_3_end <- valid_cleavage_site$position
    ###Process UTR sites data###
    external_five_UTR$chr <- apply(external_five_UTR,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
    external_three_UTR$chr <- apply(external_three_UTR,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
    valid_external_five_UTR <- subset(external_five_UTR,chr %in% as.character(1:5) & id %in% valid_transcript_id)
    valid_external_three_UTR <- subset(external_three_UTR,chr %in% as.character(1:5) & id %in% valid_transcript_id)
    ###Write data##
    write.table(valid_official_araport11_coding,valid_official_araport11_coding_path,sep='\t',quote=F,row.names=F)
    valid_official_araport11_coding_bed = valid_official_araport11_coding[c('chr','start','end','id','score','strand',
                                                                            'orf_start','orf_end','rgb','count',
                                                                            'block_size','block_related_start')]
    write_bed12(valid_official_araport11_coding_bed,valid_official_araport11_coding_bed_path)
    write.table(valid_gro,valid_gro_path,sep='\t',quote=F,row.names=F)
    write.table(valid_cleavage_site,valid_cleavage_site_path,sep='\t',quote=F,row.names=F)
    write.table(valid_external_five_UTR,valid_external_five_UTR_path,sep='\t',quote=F,row.names=F)
    write.table(valid_external_three_UTR,valid_external_three_UTR_path,sep='\t',quote=F,row.names=F)
    write.table(id_convert,id_convert_path,sep='\t',quote=F,col.names=F)
}
