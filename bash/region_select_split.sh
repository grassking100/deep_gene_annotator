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
 echo "    -s  <bool>    Split training and validation dataset with strand"
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

on_single_strand_data_command="${command} -o $saved_root/split_without_strand/single_strand_data"
on_double_strand_data_command="${command} -d  -o $saved_root/split_without_strand/double_strand_data"

echo $on_single_strand_data_command
bash $on_single_strand_data_command
echo $on_double_strand_data_command
bash $on_double_strand_data_command

if $split_with_strand && [ ! "$fold_num" ]; then
    split_with_strand_command="${command} -s  -o $saved_root/split_with_strand/single_strand_data"
    echo $split_with_strand_command
    bash $split_with_strand_command
fi
