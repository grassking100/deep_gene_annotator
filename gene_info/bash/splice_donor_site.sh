if (( $# != 3 )); then
    echo "Usage:"
    echo "    bash splice_donor_site.sh BEDFILES upstream downstream"
    exit 1
fi

bed_file="${1%.*}"
upstream=$2
downstream=$3

awk -F'\t' -v OFS="\t"  '
                         {   
                             n = split($11, sizes, ",")
                             transcrtipt_start = $2 + 1
                             split($12, related_starts, ",")
                             for (i = 1; i <= n;i++)
                             {   
                                 size = sizes[i]
                                 related_start = related_starts[i]
                                 related_end = related_starts[i] + size - 1
                                 splice_donor_site=-1
                                 if($6=="-")
                                 {
                                     if(related_start>0)
                                     {
                                         splice_donor_site = related_start
                                     }
                                 }
                                 else
                                 {
                                     if((related_end+$2) < ($3-1))
                                     {
                                         splice_donor_site = related_end       
                                     }
                                 }
                                 site = splice_donor_site+transcrtipt_start
                                 if($6=="+")
                                 {
                                     start = site-'$upstream' + 1
                                     end = site+'$downstream' + 1 
                                 }
                                 else
                                 {
                                     end = site+'$upstream' - 1 
                                     start = site-'$downstream' - 1
                                 }
                                 if(splice_donor_site!=-1 && start>=0)
                                 {
                                     print($1,start-1,end,$4,$5,$6)
                                 }
                             }
                         }'  "$bed_file.bed"
       