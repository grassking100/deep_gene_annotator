left=1000
right=500
date_=$(date +"%Y_%m_%d")
coordinate_consist_with_gene_id=coordinate_consist_with_gene_id
coordinate_consist=coordinate_consist
raw_mRNA=filtered_coding_gene.bed
exon_file=~/../home/io/Arabidopsis_thaliana/data/Araport11_coding_exon_2019_04_07

mkdir data

Rscript --save --restore script/R/bed_repair.R $raw_mRNA ${coordinate_consist}.bed all_coding_mRNA.bed
bash script/bash/nonoverlap_filter.sh ${coordinate_consist_with_gene_id}.bed

Rscript --save --restore script/R/want_unwant_bed_create_by_id.R ${coordinate_consist_with_gene_id}_nonoverlap_id.txt \
 ${coordinate_consist}.bed all_coding_mRNA.bed data/want_${date_}.bed data/unwant_${date_}.bed false
saved_num=-1
index=0
want_bed=data/want_${date_}
unwant_bed=data/unwant_${date_}
fai=~/../home/io/Arabidopsis_thaliana/data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta.fai

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
        ${want_bed}.bed all_coding_mRNA.bed \
        data/want_${date_}_iter_${index}.bed \
        data/unwant_${date_}_iter_${index}.bed \
        true
        want_bed=data/want_${date_}_iter_${index}
        unwant_bed=data/unwant_${date_}_iter_${index} 
    fi
    saved_num=$num
done
#Export to ${exon_file}_merged_with_coordinate_file.bed
bash script/bash/merge_exon_gene_by_id.sh ${exon_file}.bed data/want_${date_}_iter_${index}.bed

Rscript --save --restore script/R/merge_bed.R ${exon_file}_merged_with_coordinate_file.bed ${exon_file}_merged_with_coordinate_file_megred_exon.bed
