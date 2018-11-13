bed="${1%.*}"
bedtools sort -i "$bed.bed" > "${bed}_sorted.bed"
bedtools merge -s -c 4,5,6 -o distinct,distinct,distinct -delim "_" -i "${bed}_sorted.bed"  -v > "${bed}_merged_temp.bed"
awk -F'\t' -v OFS="\t"  '{print $1,$2,$3,$5,$6,$7;}'  "${bed}_merged_temp.bed" > "${bed}_nonoverlap.bed"

rm "${bed}_merged_temp.bed"
rm "${bed}_sorted.bed"