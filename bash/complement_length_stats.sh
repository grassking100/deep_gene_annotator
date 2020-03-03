#!/bin/bash
## function print usage
usage(){
 echo "Usage: The script calculated length statistic data on complemented data"
 echo "  Arguments:"
 echo "    -i  <string>  Path of bed"
 echo "    -f  <string>  Path of genome fai"
 echo "    -o  <string>  Path of output"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash bed_analysis.sh -i example.bed -f genome.fasta -o complemented_length.stats"
 echo ""
}

while getopts i:f:o:h option
 do
  case "${option}"
  in
   i )input_path=$OPTARG;;
   f )fai_path=$OPTARG;;
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
if [ ! "$fai_path" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi
if [ ! "$output_path" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess

input_file="${input_path%.*}"
fai_file="${fai_path%.*}"

bedtools sort -i $input_path > ${input_file}_sorted.bed
awk -F'\t' -v OFS="\t" '{print($1,$2)}' $fai_path > $fai_file.genome
bedtools complement -i ${input_file}_sorted.bed -g $fai_file.genome > ${input_file}_complement.bed
python3 $preprocess_main_root/bed_length_stats.py -i ${input_file}_complement.bed -o $output_path

rm -rf ${input_file}_sorted.bed
rm -rf $fasta_file.genome
rm -rf ${input_file}_complement.bed

