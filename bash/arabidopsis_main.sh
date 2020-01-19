#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating annotation data"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -r  <string>  Directory of Arabidopsis thaliana data"
 echo "    -o  <string>  Directory of output folder"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -m  <bool>    Merge regions which are overlapped                 [default: false]"
 echo "    -c  <bool>    Remove gene with altenative donor site and accept site    [default: false]"
 echo "    -i  <bool>    Merge regions which inner singal                [default: false]"
 echo "    -x  <bool>    Remove gene with non-coding transcript    [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_main.sh -u 10000 -d 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:r:o:s:mcxih option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   r )root=$OPTARG;;
   o )saved_root=$OPTARG;;
   s )source_name=$OPTARG;;
   m )merge_overlapped=true;;
   c )remove_alt_site=true;;
   x )remove_non_coding=true;;
   i )remove_inner_end=true;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$upstream_dist" ]; then
    echo "Missing option -u"
    usage
    exit 1
fi

if [ ! "$downstream_dist" ]; then
    echo "Missing option -w"
    usage
    exit 1
fi

if [ ! "$root" ]; then
    echo "Missing option -r"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$source_name" ]; then
    echo "Missing option -s"
    usage
    exit 1
fi

if [ ! "$merge_overlapped" ]; then
    merge_overlapped=false
fi

if [ ! "$remove_alt_site" ]; then
    remove_alt_site=false
fi

if [ ! "$remove_non_coding" ]; then
    remove_non_coding=false
fi

if [ ! "$remove_inner_end" ]; then
    remove_inner_end=false
fi

#Set parameter
genome_path=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
preprocessed_root=$saved_root/preprocessed
processed_root=$saved_root/processed
id_convert_table_path=$preprocessed_root/id_convert.tsv
processed_bed_path=$preprocessed_root/processed.bed
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess

mkdir -p $saved_root

command="$bash_root/arabidopsis_data_prepair.sh -u $upstream_dist -d $downstream_dist -r $root -o $preprocessed_root -s $source_name"
if $remove_inner_end; then
    command="${command} -i"
fi
echo $command
bash $command

if [ ! -e "$processed_root/result/selected_region.bed" ]; then
    command="$bash_root/process_data.sh -u $upstream_dist -d $downstream_dist -g $genome_path -i $preprocessed_root/coordinate_consist.bed -o $processed_root -s $source_name -t $id_convert_table_path -b $processed_bed_path"

    if $merge_overlapped; then
        command="${command} -m"
    fi
    
    if $remove_alt_site; then
        command="${command} -c"
    fi
    
    if $remove_non_coding; then
        command="${command} -x"
    fi
    
    echo $command
    bash $command
else
    echo "The program process_data.sh is skipped"
fi

bash $bash_root/region_select_split.sh -g $genome_path -t $preprocessed_root/id_convert.tsv -p $processed_root -o $processed_root/split -s
