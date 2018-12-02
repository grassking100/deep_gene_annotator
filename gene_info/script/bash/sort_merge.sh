bed="${1%.*}"
bedtools sort -i "$bed.bed" | bedtools merge -s -c 4,5,6 -o distinct,distinct,distinct -delim ","  | awk -F'\t' -v OFS="\t"  '{print $1,$2,$3,$5,$6,$7;}' > "${bed}_merged.bed"
