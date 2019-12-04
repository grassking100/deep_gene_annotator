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

bedtools sort -i $region_path > _region_sorted.temp
bedtools sort -i $data_path > _data_sorted.temp

awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,$4,$5,$6)
}' _region_sorted.temp > _region_sorted.simple

if $strand; then
    bedtools map -a _region_sorted.simple -b _data_sorted.temp -c 4,4 -o count_distinct,distinct -s > _temp.bed
else
    bedtools map -a _region_sorted.simple -b _data_sorted.temp -c 4,4 -o count_distinct,distinct > _temp.bed
fi
    
if [  "$single_count_path" ]; then
    awk -F '\t' -v OFS='\t' '{print($1,$2,$3,$8,$5,$6,$7)}' _temp.bed > $single_count_path
fi

awk -F '\t' -v OFS='\t' '{print($1,$2,$3,$8,$5,$6)}' _temp.bed > $output_path

rm _temp.bed
rm _region_sorted.simple
rm _region_sorted.temp
rm _data_sorted.temp
