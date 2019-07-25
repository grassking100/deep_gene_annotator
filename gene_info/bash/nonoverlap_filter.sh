#!/bin/bash
## function print usage
usage(){
 echo "Usage: Get nonoverlap region id"
 echo "  Arguments:"
 echo "    -i  <string>  Input path"
 echo "    -o  <string>  Output path"
 echo "  Options:"
 echo "    -h  Print help message and exit"
 echo "Example: bash nonoverlap_filter.sh -i raw.bed -o nonoverlap_id.txt"
 echo ""
}
while getopts i:o:h option
 do
  case "${option}"
  in
   i )input_path=$OPTARG;;
   o )output_path=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$input_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi
if [ ! "$output_path" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash $script_root/sort_merge.sh -i $input_path -o _temp.bed
awk -F '\t' -v OFS='\t' '{
    n = split($4, ids, ",")
    if(n==1)
    {
        print(ids[1])
    }
}' _temp.bed > $output_path
rm _temp.bed
