#!/bin/bash
## function print usage
usage(){
 echo "Usage: Get regions which each strands have at most one gene"
 echo "  Arguments:"
 echo "    -i  <string>  Input region gff path"
 echo "    -d  <string>  Input data gff path"
 echo "    -o  <string>  Output path"
 echo "  Options:"
 echo "    -h  Print help message and exit"
 echo "Example: bash single_gene_region_filter.sh -i region.bed -d data.bed -o nonoverlap_id.txt"
 echo ""
}
while getopts i:d:o:h option
 do
  case "${option}"
  in
   i ) region_path=$OPTARG;;
   d ) data_path=$OPTARG;;
   o ) output_path=$OPTARG;;
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
    echo "Missing option -r"
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

script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bedtools sort -i $region_path > $region_path.temp
bedtools sort -i $data_path > $data_path.temp

awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,$4,$5,$6)
}' $region_path.temp > $region_path.simple.temp

bedtools map -a $region_path.simple.temp -b $data_path.temp -c 4 -o count_distinct -s > _temp.bed
    
awk -F '\t' -v OFS='\t' '{
    n = $7
    if(n<=1)
    {
        print($1,$2,$3,$4,$5,$6)
    }
}' _temp.bed > _single.bed

bedtools map -a _single.bed -b $data_path.temp -c 4 -o count_distinct > _single2.bed

awk -F '\t' -v OFS='\t' '{
    n = $7
    if(n<=2)
    {
        print($1,$2,$3,$4,$5,".")
    }
}' _single2.bed > _single3.bed

bash $script_root/sort_merge.sh -i _single3.bed -o $output_path

rm _temp.bed
rm _single.bed
rm _single2.bed
rm _single3.bed
rm $region_path.simple.temp
rm $region_path.temp
rm $data_path.temp
