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
 echo "    -s  <bool>    Split training and validation dataset with and without strand (default: without strand only)"
 echo "    -n  <int>     Fold number (exclude testing dataset), if it is not provided then it would decided by chromosome number"
 echo "    -h            Print help message and exit"
 echo "Example: bash region_select_split.sh -p /home/io/process -t /home/io/preprocess/id_convert.tsv -o ./data/2019_07_12 -g genome.fasta"
 echo ""
}

while getopts g:t:p:o:n:sh option
 do
  case "${option}"
  in
   g )genome_path=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   p )processed_root=$OPTARG;;
   o )saved_root=$OPTARG;;
   n )fold_num=$OPTARG;;
   s )split_with_strand=true;;
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

if [ ! "$split_with_strand" ]; then
    split_with_strand=false
fi

#Set parameter
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
mkdir -p $saved_root

command="$bash_root/split.sh -g $genome_path -p $processed_root -t $id_convert_table_path"

if [ "$fold_num" ]; then
    command="${command} -n $fold_num"
fi

single_strand_split_without_strand_path=$saved_root/single_strand_data/split_without_strand

rm -rf $single_strand_split_without_strand_path

on_single_strand_data_command="${command} -o $single_strand_split_without_strand_path"

echo $on_single_strand_data_command
bash $on_single_strand_data_command

if $split_with_strand && [ ! "$fold_num" ]; then
    single_strand_split_with_strand_path=$saved_root/single_strand_data/split_with_strand
    rm -rf $single_strand_split_with_strand_path
    on_single_strand_data_command="${command} -s  -o $saved_root/single_strand_data/split_with_strand"
    echo $on_single_strand_data_command
    bash $on_single_strand_data_command
fi
