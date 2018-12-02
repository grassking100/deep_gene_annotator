library("gsubfn")
source('script/R/gff.R')
source('script/R/utils.R')
Araport11_GFF3_genes_transposons=read_gff('data/Araport11_GFF3_genes_transposons.201606.gff')

Araport11_gene_gff <- Araport11_GFF3_genes_transposons[Araport11_GFF3_genes_transposons$feature == 'gene',]
Araport11_mRNA_gff <- Araport11_GFF3_genes_transposons[Araport11_GFF3_genes_transposons$feature == 'mRNA',]
Araport11_exon_gff <- Araport11_GFF3_genes_transposons[Araport11_GFF3_genes_transposons$feature %in% c('three_prime_UTR','five_prime_UTR','CDS'),]
Araport11_three_prime_UTR_gff <- Araport11_GFF3_genes_transposons[Araport11_GFF3_genes_transposons$feature %in% c('three_prime_UTR'),]
Araport11_five_prime_UTR_gff <- Araport11_GFF3_genes_transposons[Araport11_GFF3_genes_transposons$feature %in% c('five_prime_UTR'),]

write_bed(gff_to_bed(Araport11_gene_gff,'id'),str_with_time('Araport11_gene_','.bed'))
write_bed(gff_to_bed(Araport11_mRNA_gff,'id'),str_with_time('Araport11_mRNA_','.bed'))
write_bed(gff_to_bed(Araport11_exon_gff,'parent'),str_with_time('Araport11_exon_','.bed'))
write_bed(gff_to_bed(Araport11_three_prime_UTR_gff,'parent'),str_with_time('Araport11_three_prime_UTR_','.bed'))
write_bed(gff_to_bed(Araport11_five_prime_UTR_gff,'parent'),str_with_time('Araport11_five_prime_UTR_','.bed'))
remove(Araport11_GFF3_genes_transposons)
remove(Araport11_gene_gff)
remove(Araport11_mRNA_gff)
remove(Araport11_exon_gff)
remove(Araport11_three_prime_UTR_gff)
remove(Araport11_five_prime_UTR_gff)
