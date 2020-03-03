#!/bin/bash
## function print usage
usage(){
 echo "Usage: Get genes partialy mapping to region"
 echo "  Arguments:"
 echo "    -i  <string>  Input region bed path"
 echo "    -d  <string>  Input gene bed path"
 echo "    -o  <string>  Output path"
 echo "  Options:"
 echo "    -c  <string>  Path to write regions which contain genes number at single strand"
 echo "    -s  Consider strand"
 echo "    -h  Print help message and exit"
 echo "Example: bash gene_to_region_mapping.sh -i region.bed -d gene.bed -o filter.bed"
 echo ""
}
while getopts i:d:o:c:sh option
 do
  case "${option}"
  in
   i ) region_path=$OPTARG;;
   d ) data_path=$OPTARG;;
   o ) output_path=$OPTARG;;
   c ) single_count_path=$OPTARG;;
   s ) strand=true;;
   h ) usage; exit 1;;
   : ) echo "Option $OPTARG requires an argument"
       usage; exit 1
       ;;
   \?) echo "Invalid option: $OPTARG"
       usage; exit 1
       ;;
 esac
done

if [ ! "$region_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi

if [ ! "$data_path" ]; then
    echo "Missing option -d"
    usage
    exit 1
fi

if [ ! "$output_path" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$strand" ]; then
    strand=false
fi


script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

region_file="${region_path%.*}"
data_file="${data_path%.*}"

bedtools sort -i $region_path > ${region_file}_sorted.bed
bedtools sort -i $data_path > ${data_file}_sorted.bed

awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,$4,$5,$6)
}' ${region_file}_sorted.bed > ${region_file}_sorted_simple.bed

if $strand; then
    bedtools map -a ${region_file}_sorted_simple.bed -b ${data_file}_sorted.bed -c 4,4 -o count_distinct,distinct -s > ${region_file}_mapped.bed
else
    bedtools map -a ${region_file}_sorted_simple.bed -b ${data_file}_sorted.bed -c 4,4 -o count_distinct,distinct > ${region_file}_mapped.bed
fi
    
if [  "$single_count_path" ]; then
    awk -F '\t' -v OFS='\t' '{print($1,$2,$3,$8,$5,$6,$7)}' ${region_file}_mapped.bed > $single_count_path
fi

awk -F '\t' -v OFS='\t' '{print($1,$2,$3,$8,$5,$6)}' ${region_file}_mapped.bed > $output_path

rm ${region_file}_mapped.bed
rm ${region_file}_sorted_simple.bed
rm ${region_file}_sorted.bed
rm ${data_file}_sorted.bed
