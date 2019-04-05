left=1000
right=500
date_=$(date +"%Y_%m_%d")
coordinate_consist_with_gene_id=coordinate_consist_with_gene_id_2019_04_04
coordinate_consist=coordinate_consist_with_2019_04_04
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
bash script/bash/nonoverlap_filter.sh ${coordinate_consist_with_gene_id}.bed
#Read Araport11_mRNA_${time}.bed and coordinate_consist_with_gene_id_${time}_nonoverlap_id.txt
#Write to nonoverlap_coordinate_consist_${time}.bed
#Write to unwant_mRNA_${time}.bed
Rscript --save --restore script/R/want_unwant_bed_create_by_id.R ${coordinate_consist_with_gene_id}_nonoverlap_id.txt \
 ${coordinate_consist}.bed  Araport11_selected_mRNA_2018_11_12.bed want_${date_}.bed unwant_${date_}.bed false
saved_num=-1
index=0
want_bed=want_${date_}
unwant_bed=unwant_${date_}
fai=araport_11_Arabidopsis_thaliana_Col-0_rename.fasta.fai

while true;do
    
    bash script/bash/danger_filter.sh ${want_bed}.bed ${unwant_bed}.bed ${fai} ${left} ${right}
    num=$( cat ${want_bed}_expand_left_${left}_right_${right}_safe_zone_id.txt | wc -l)
    echo $num
    if (($num==$saved_num)) ;then
        break
    else
        index=$((index+1))    
        Rscript --save --restore script/R/want_unwant_bed_create_by_id.R \
        ${want_bed}_expand_left_${left}_right_${right}_safe_zone_id.txt \
        ${want_bed}.bed Araport11_selected_mRNA_2018_11_12.bed \
        want_${date_}_iter_${index}.bed \
        unwant_${date_}_iter_${index}.bed \
        true
        want_bed=want_${date_}_iter_${index}
        unwant_bed=unwant_${date_}_iter_${index} 
    fi
    saved_num=$num
done
exon_file=Araport11_selected_exon_2018_11_12
#Export to ${exon_file}_merged_with_coordinate_file.bed
bash script/bash/merge_exon_gene_by_id.sh ${exon_file}.bed want_${date_}_iter_${index}.bed

Rscript --save --restore script/R/merge_bed.R ${exon_file}_merged_with_coordinate_file.bed ${exon_file}_merged_with_coordinate_file_megred_exon.bed
