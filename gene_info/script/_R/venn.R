args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=3) {
  stop("Three argument must be supplied (input file)", call.=FALSE)
}

gro <- args[1]
cleavage <- args[1]
selected <- args[3]
gro <- read.table(gro,sep='\t')
cleavage <- read.table(cleavage,sep='\t')
selected <- read_bed12(selected)

venn=venn.diagram(list(exist_gro_sites=gro$gene_id,
                       exist_cleavage_sites=cleavage$gene_id,
                       train_gene=selected$gene_id,
                       araport11=biomart_araport11_coding$gene_id),
                       filename = NULL,force.unique=T)
grid.draw(venn)
 