mRNA_file=$1
exon_file=$2
coordinate_file=$3
mRNA_file="${mRNA_file%.*}"
exon_file="${exon_file%.*}"

bash sort_merge.sh "$mRNA_file.bed"

bedtools subtract -s -a "$coordinate_file" -b "${mRNA_file}_merged.bed" > "${mRNA_file}_extended_region.bed"

bedtools intersect -s -a $coordinate_file  -b "$exon_file.bed" > "${exon_file}_intersected.bed"
cat "${exon_file}_intersected.bed" "${mRNA_file}_extended_region.bed" > "${exon_file}_with_extend.bed"
bedtools sort -i "${exon_file}_with_extend.bed" > "${exon_file}_with_extend_sorted.bed"
#bedtools merge -s -c 4 -o distinct -delim "_" -i "${exon_file}_with_extend_sorted.bed"  > "${exon_file}_extended_temp.bed"
sort -k4,4  "${exon_file}_with_extend_sorted.bed" |bedtools groupby -g 4 -c 1,2,3,5,6,1 -o distinct,collapse,collapse,distinct,distinct,count > ${five_utr_file}_merged.bed

#
#awk -F'\t' -v OFS="\t"  '{print $1,$2,$3,$5,".",$4;}'  "${exon_file}_extended_temp.bed" > "${exon_file}_extended.bed"
#bedtools intersect -s -a $coordinate_file -b "${exon_file}_extended.bed" -wa -wb  | cut -f1,2,3,4,5,6,7,8,10,11 > temp.bed
#awk -F'\t' -v OFS="\t"  '{print $1,$2,$3,$4,$5,$6,$7,$9-$2,$10-$9,$7,$8;}'  temp.bed> temp2.bed
#sort -k 1,1 -k 2,2n -k 3,3n -k 6,6n -k 7,7n  temp2.bed > temp3.bed 
#bedtools merge -s  -i temp3.bed -c 4,5,7,8,9,10,11 -o distinct,distinct,count,collapse,collapse,distinct,distinct > temp4.bed
#awk -F'\t' -v OFS="\t"  '{print $1,$2,$3,$5,$6,$4,$10,$11,".",$7,$9,$8;}'  temp4.bed > "${exon_file}_with_mRNA.bed"

rm "${exon_file}_intersected.bed"
rm "${exon_file}_with_extend.bed"
rm "${exon_file}_extended_temp.bed"
rm "${exon_file}_with_extend_sorted.bed"
rm temp.bed
rm temp2.bed
rm temp3.bed
rm temp4.bed