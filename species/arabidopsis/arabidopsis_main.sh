#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating arabidopsis annotation data"
 echo "  Arguments:"
 echo "    -i  <string>  Directory of Arabidopsis thaliana data"
 echo "    -o  <string>  Directory of output folder"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -r  <string>  Additinoal keyword args for coordinate_redefined.sh"
 echo "    -f  <string>  Additinoal keyword args for filter_bed.sh"
 echo "    -m  <string>  Additinoal keyword args for process_data.sh"
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_main.sh -u 10000 -d 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:i:o:s:f:r:m:h option
do
    case "${option}"
    in
    u )upstream_dist=$OPTARG;;
    d )downstream_dist=$OPTARG;;
    i )root=$OPTARG;;
    o )saved_root=$OPTARG;;
    s )source_name=$OPTARG;;
    r )coordinate_redefined_kwargs=$OPTARG;;
    f )filter_kwargs=$OPTARG;;
    m )process_data_kwargs=$OPTARG;;
    h )usage; exit 1;;
    : )echo "Option $OPTARG requires an argument"
       usage; exit 1;;
    \?)echo "Invalid option: $OPTARG"
       usage; exit 1;;
    esac
done

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

if [ ! "$root" ]; then
    echo "Missing option -i"
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

mkdir -p $saved_root

echo "name,value" > $saved_root/main_kwargs.csv
echo "root,$root" >> $saved_root/main_kwargs.csv
echo "saved_root,$saved_root" >> $saved_root/main_kwargs.csv
echo "upstream_dist,$upstream_dist" >> $saved_root/main_kwargs.csv
echo "downstream_dist,$downstream_dist" >> $saved_root/main_kwargs.csv
echo "source_name,$source_name" >> $saved_root/main_kwargs.csv
echo "coordinate_redefined_kwargs,$coordinate_redefined_kwargs" >> $saved_root/main_kwargs.csv
echo "filter_kwargs,$filter_kwargs" >> $saved_root/main_kwargs.csv
echo "process_data_kwargs,$process_data_kwargs" >> $saved_root/main_kwargs.csv

#Set parameter
arabidopsis_util_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
src_root=$arabidopsis_util_root/../..
bash_root=$src_root/bash
preprocess_main_root=$src_root/sequence_annotation/preprocess
preprocessed_root=$saved_root/preprocessed
id_convert_table_path=$preprocessed_root/id_convert.tsv
preserved_bed_path=$preprocessed_root/consistent.bed
background_bed_path=$preprocessed_root/processed.bed
tss_gff_path=$preprocessed_root/tss.gff3
cs_gff_path=$preprocessed_root/cleavage_site.gff3
genome_path=$preprocessed_root/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
non_hypothetical_gene_id_path=$preprocessed_root/non_hypothetical_gene_id.txt

if [ ! -e "$preprocessed_root/consistent.bed" ]; then
    bash $arabidopsis_util_root/prepair_data.sh -r $root -o $preprocessed_root
fi

command="$bash_root/pipeline.sh -p $preserved_bed_path -b $background_bed_path -t $tss_gff_path -c $cs_gff_path -g $genome_path -i $id_convert_table_path -u $upstream_dist -d $downstream_dist -o $saved_root -s $source_name"

if [ "$filter_kwargs" ]; then
    command="$command -f \"$filter_kwargs\""
fi

if [ "$coordinate_redefined_kwargs" ]; then
    command="$command -r \"$coordinate_redefined_kwargs\""
fi

if [ "$process_data_kwargs" ]; then
    command="$command -m \"$process_data_kwargs\""
fi

echo $command
eval "bash $command"
