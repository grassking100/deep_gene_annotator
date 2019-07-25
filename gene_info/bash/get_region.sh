#!/bin/bash
## function print usage
usage(){
 echo "Usage: Get upstream and donwstream region"
 echo "  Arguments:"
 echo "    -i  <string>  Input path"
 echo "    -o  <string>  Output path"
 echo "    -f  <string>  fai file path"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "  Options:"
 echo "    -h  Print help message and exit"
 echo "Example: bash get_region.sh -i raw.bed -o result.bed -u 100 -d 100 -f genome.fai"
 echo ""
}
while getopts i:o:u:d:f:h option
 do
  case "${option}"
  in
   i )input_path=$OPTARG;;
   o )output_path=$OPTARG;;
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   f )fai_path=$OPTARG;;
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
if [ ! "$upstream_dist" ]; then
    echo "Missing option -u"
    usage
    exit 1
fi
if [ ! "$downstream_dist" ]; then
    echo "Missing option -d"
    usage
    exit 1
fi
if [ ! "$fai_path" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi

script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bedtools slop -s -i $input_path -g $fai_path -l $upstream_dist -r $downstream_dist > _temp.bed
bash $script_root/sort_merge.sh -i _temp.bed -o $output_path
rm _temp.bed
