if (( $# != 2 )); then
    echo "Usage:"
    echo "    bash cleavage_site.sh BEDFILES RADIUS"
    exit 1
fi
input_path=$1
radius=$2

awk -F'\t' -v OFS="\t"  '
                         {   
                             if($6=="+")
                             {
                                 start = $3 - '$radius'
                                 end = $3 + '$radius'
                             }
                             else
                             {
                                 start = $2 + 1 - '$radius'
                                 end = $2 + 1 + '$radius'
                             }
                             if(start>=1)
                             {
                                 print($1,start-1,end,$4,$5,$6)
                             }
                         }'  $input_path 
