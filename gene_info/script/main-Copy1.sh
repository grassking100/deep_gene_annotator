left=1000
right=500
date_=$(date +"%Y_%m_%d")
coordinate_consist_with_gene_id=data/2018_11_27/coordinate_consist_with_gene_id_2018_11_27
coordinate_consist=data/2018_11_27/coordinate_consist_with_2018_11_27
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
#Write nonoverlap gene id to coordinate_consist_with_gene_id_${time}_nonoverlap_id.txt
#bash script/bash/nonoverlap_filter.sh ${coordinate_consist_with_gene_id}.bed
#Read Araport11_mRNA_${time}.bed and coordinate_consist_with_gene_id_${time}_nonoverlap_id.txt
#Write to nonoverlap_coordinate_consist_${time}.bed
#Write to unwant_mRNA_${time}.bed
#Rscript --save --restore script/R/want_unwant_bed_create_by_id.R ${coordinate_consist_with_gene_id}_nonoverlap_id.txt \
# ${coordinate_consist}.bed  data/Araport11_mRNA_${date_}.bed data/want_${date_}.bed data/unwant_${date_}.bed false
saved_num=-1
index=0
want_bed=data/want_${date_}
unwant_bed=data/unwant_${date_}
fai=data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta.fai

