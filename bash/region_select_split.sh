#!/bin/bash
## function print usage
usage(){
 echo "Usage: The program executes split.sh"
 echo "  Arguments:"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -t  <string>  Path of id_convert_table"
 echo "    -p  <string>  Path of processed folder"
 echo "    -o  <string>  Path of output folder"
 echo "  Options:"
 echo "    -d  <bool>    Split on double-strand data"
 echo "    -s  <bool>    Each strand on training and validation dataset would be treat independently, if data be split is double-strand data then it would be ignore"
 echo "    -n  <int>     Fold number (exclude testing dataset), if it is not provided then it would decided by chromosome number"
 echo "    -h            Print help message and exit"
 echo "Example: bash region_select_split.sh -p /home/io/process -t /home/io/preprocess/id_convert.tsv -o ./data/2019_07_12 -g genome.fasta"
 echo ""
}

while getopts g:t:p:o:n:sdnh option
 do
  case "${option}"
  in
   g )genome_path=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   p )processed_root=$OPTARG;;
   o )saved_root=$OPTARG;;
   n )fold_num=$OPTARG;;
   d )on_double_strand_data=true;;
   s )treat_strand_independent=true;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$genome_path" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi

if [ ! "$id_convert_table_path" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

if [ ! "$processed_root" ]; then
    echo "Missing option -p"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$on_double_strand_data" ]; then
    on_double_strand_data=false
fi

if [ ! "$treat_strand_independent" ]; then
    treat_strand_independent=false
fi

#Set parameter
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
region_path=$processed_root/result/region_id_conversion.tsv
mkdir -p $saved_root

command="$bash_root/split.sh -g $genome_path -p $processed_root -t $id_convert_table_path -r $region_path"

if [ "$fold_num" ]; then
    command="${command} -n $fold_num"
fi

if $on_double_strand_data; then
    command="${command} -d"
    result_root=$saved_root/double_strand_data
else
    result_root=$saved_root/single_strand_data
fi

if $treat_strand_independent; then
    command="${command} -s"
    result_root=$result_root/split_with_strand
else
    result_root=$result_root/split_without_strand
fi

#rm -rf $result_root
command="${command} -o $result_root"

echo $command
bash $command
