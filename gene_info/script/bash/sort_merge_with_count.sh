bed="${1%.*}"
bedtools sort -i "$bed.bed" | bedtools merge -s -c 4,5,6,5 -o distinct,distinct,distinct,count -delim "," > "${bed}_merged_with_count.bed"
