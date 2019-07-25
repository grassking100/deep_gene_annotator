if (( $# != 2 )); then
    echo "Usage:"
    echo "    bash transcription_start_site.sh BEDFILES RADIUS"
    exit 1
fi
bed_file=$1
radius=$2
awk -F'\t' -v OFS="\t"  '
                         {   
                             if($6=="+")
                             {
                                 start = $2 + 1 - '$radius'
                                 end = $2 + 1 + '$radius'
                             }
                             else
                             {
                                 start = $3 - '$radius'
                                 end = $3 + '$radius'
                             }
                             if(start>=1)
                             {
                                 print($1,start-1,end,$4,$5,$6)
                             }
                         }'  $bed_file
