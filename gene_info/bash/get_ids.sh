bed="${1%.*}"
awk -F '\t' -v OFS='\t' '{
    n = split($4, ids, ",")
    for(i=1;i<=n;i++)
    {
        print(ids[i])
    }
}' "${bed}.bed" > "${bed}_id.txt"