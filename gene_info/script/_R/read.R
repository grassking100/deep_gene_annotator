print("Read")
biomart_araport_11_gene_info <- read.csv('~/../home/io/Arabidopsis_thaliana/data/biomart_araport_11_gene_info_2018_11_27.csv',stringsAsFactors=F)
biomart_araport_11_gene_info = biomart_araport_11_gene_info[c('Gene.stable.ID',
                                                              'Gene.start..bp.',
                                                              'Gene.end..bp.',
                                                              'Transcript.stable.ID',
                                                              'Transcript.start..bp.','Transcript.end..bp.',
                                                              'Strand','Chromosome.scaffold.name','Transcript.type')]
names(biomart_araport_11_gene_info) <- c('gene_id','biomart_gene_start','biomart_gene_end',
                                         'transcript_id','biomart_transcript_start',
                                         'biomart_transcript_end','strand','chr','transcript_type')
biomart_araport_11_gene_info <- subset(biomart_araport_11_gene_info,chr %in% as.character(1:5))

id_convert <- as.character(biomart_araport_11_gene_info$gene_id)
names(id_convert) <- as.character(biomart_araport_11_gene_info$transcript_id)


official_araport11_coding <-read_bed12('~/../home/io/Arabidopsis_thaliana/data/Araport11_GFF3_genes_transposons.201606_coding_repair_2019_04_07.bed')

official_araport11_coding$chr <- apply(official_araport11_coding,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
official_araport11_coding <- subset(official_araport11_coding,chr %in% as.character(1:5))
biomart_araport11_coding <- subset(biomart_araport_11_gene_info,transcript_type == 'protein_coding')
valid_transcript_id <- intersect(official_araport11_coding$id,biomart_araport11_coding$transcript_id)
valid_gene_id <- id_convert[as.character(valid_transcript_id)]

gro_1 <- read.csv('~/../home/io/Arabidopsis_thaliana/data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv',
                  sep='\t',header=T,stringsAsFactors=F,comment.char = "#")
gro_2 <- read.csv('~/../home/io/Arabidopsis_thaliana/data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv',
                  sep='\t',header=T,stringsAsFactors=F,comment.char = "#")

gro_ <- rbind(gro_1,gro_2)
temp <- gro_[c('chr','start','end','strand')]
gro <- gro_[duplicated(temp),]
gro <- gro[c('chr','strand','Normalized.Tag.Count','start','end')]
names(gro) <- c('chr','strand','tag_count','start','end')
gro['mode'] <- round((gro$end + gro$start)/2)

cleavage_site <- read.csv('~/../home/io/Arabidopsis_thaliana/data/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv',
                          stringsAsFactors=F)

cleavage_site[cleavage_site$Strand=='fwd',]['Strand'] <- '+'
cleavage_site[cleavage_site$Strand=='rev',]['Strand'] <- '-'
cleavage_site <- cleavage_site[,c('Chromosome','Strand','Position','Raw.DRS.read.count')]
colnames(cleavage_site) <- c('chr','strand','position','read_count')
cleavage_site$chr <- apply(cleavage_site,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})

external_five_UTR <- read_bed('~/../home/io/Arabidopsis_thaliana/data/most_five_UTR_2019_04_28.bed')
external_three_UTR <- read_bed('~/../home/io/Arabidopsis_thaliana/data/most_three_UTR_2019_04_28.bed')

#external_five_UTR$chr <- apply(external_five_UTR,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})
#external_three_UTR$chr <- apply(external_three_UTR,1,function(x) {substr(x['chr'],start=4,stop=nchar(x['chr']))})

valid_external_five_UTR <- subset(external_five_UTR,chr %in% as.character(1:5) & id %in% valid_transcript_id)
valid_external_three_UTR <- subset(external_three_UTR,chr %in% as.character(1:5) & id %in% valid_transcript_id)


valid_gro <- subset(gro,chr %in% as.character(1:5))
valid_gro <- unique(valid_gro)
valid_cleavage_site <- subset(cleavage_site,chr %in% as.character(1:5))
valid_cleavage_site <- unique(valid_cleavage_site)
valid_official_araport11_coding <- subset(official_araport11_coding,id %in% valid_transcript_id)
valid_official_araport11_coding$gene_id <- id_convert[valid_official_araport11_coding$id]

orf_start_convert <- as.numeric(valid_official_araport11_coding$orf_start)
names(orf_start_convert) <- as.character(valid_official_araport11_coding$id)

orf_end_convert <- as.numeric(valid_official_araport11_coding$orf_end)
names(orf_end_convert) <- as.character(valid_official_araport11_coding$id)

valid_official_araport11_coding$exist_5_end <- get_five_end(valid_official_araport11_coding,
                                                            'start','end')
valid_official_araport11_coding$exist_3_end <- get_three_end(valid_official_araport11_coding,
                                                             'start','end')
valid_gro$evidence_5_end <- valid_gro$mode
valid_cleavage_site$evidence_3_end <- valid_cleavage_site$position

find_gene_id <- function(ids)
{
  return(id_convert[as.character(ids)])
}
find_orf_start <- function(ids)
{
  return(orf_start_convert[as.character(ids)])
}
find_orf_end<- function(ids)
{
  return(orf_end_convert[as.character(ids)])
}

remove(temp)
remove(gro_)
remove(biomart_araport_11_gene_info)

