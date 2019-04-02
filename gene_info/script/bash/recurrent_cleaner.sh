if (( $# != 10 )); then
    echo "Usage:"
    echo "    bash recurrent_cleaner.sh coordinate_consist_with_gene_id coordinate_consist mRNA_file"
    echo "                              want_bed unwant_bed fai dist_to_5 dist_to_3 separate_path result_path"
    exit 1
fi
coordinate_consist_with_gene_id=$1
coordinate_consist=$2
mRNA_file=$3
want_bed=$4
unwant_bed=$5
fai=$6
dist_to_5=$7
dist_to_3=$8
separate_path=$9
result_path=${10}

Rscript --save --restore script/R/want_unwant_bed_create_by_id.R \
${coordinate_consist_with_gene_id}_nonoverlap_id.txt \
${coordinate_consist}.bed ${mRNA_file}.bed \
${want_bed}.bed ${unwant_bed}.bed false

saved_num=-1
index=0

while true;do
    echo "Safe filter:${index}"
    bash script/bash/safe_filter.sh ${want_bed}.bed ${unwant_bed}.bed ${fai} ${dist_to_5} ${dist_to_3}
    num=$( cat ${want_bed}_expand_left_${dist_to_5}_right_${dist_to_3}_safe_zone_id.txt | wc -l)
    echo "Seqeunce number:${num}"
    if (($num==$saved_num)) ;then
        break
    else
        index=$((index+1))    
        Rscript --save --restore script/R/want_unwant_bed_create_by_id.R \
        ${want_bed}_expand_left_${dist_to_5}_right_${dist_to_3}_safe_zone_id.txt \
        ${want_bed}.bed ${mRNA_file}.bed \
        ${separate_path}/want_iter_${index}.bed \
        ${separate_path}/unwant_iter_${index}.bed \
        true
        want_bed=${separate_path}/want_iter_${index}
        unwant_bed=${separate_path}/unwant_iter_${index} 
    fi
    saved_num=$num
done

cp ${separate_path}/want_iter_${index}.bed ${result_path}/mRNA_coordinate.bed