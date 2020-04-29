#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating annotation data"
 echo "  Arguments:"
 echo "    -p  <string>  Preserved transcript data in BED format"
 echo "    -b  <string>  Background transcript data in BED format"
 echo "    -t  <string>  TSS gff path"
 echo "    -c  <string>  Cleavage site gff path"
 echo "    -g  <string>  Genome path"
 echo "    -i  <string>  Transcript and gene id conversion table"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -o  <string>  Directory of output folder"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -r  <string>  Additinoal keyword args for redefine_coordinate.sh"
 echo "    -f  <string>  Additinoal keyword args for filter_bed.sh"
 echo "    -m  <string>  Additinoal keyword args for process_data.sh"
 echo "    -h            Print help message and exit"
 echo "Example: bash pipeline.sh -u 100 -d 100 -p preserved.bed -b background.bed -o result -s temp -t tss.gff3 -c cs.gff3 -g genome.fasta -i table.tsv"
 echo ""
}

while getopts p:b:t:c:g:i:u:d:o:s:f:r:m:h option
do
    case "${option}"
    in
    p )preserved_bed_path=$OPTARG;;
    b )background_bed_path=$OPTARG;;
    t )tss_path=$OPTARG;;
    c )cleavage_site_path=$OPTARG;;
    g )genome_path=$OPTARG;;
    i )id_convert_table_path=$OPTARG;;
    u )upstream_dist=$OPTARG;;
    d )downstream_dist=$OPTARG;;
    o )saved_root=$OPTARG;;
    s )source_name=$OPTARG;;
    f )filter_kwargs=$OPTARG;;
    r )redefine_coordinate_kwargs=$OPTARG;;
    m )process_data_kwargs=$OPTARG;;
    h )usage; exit 1;;
    : )echo "Option $OPTARG requires an argument"
       usage; exit 1;;
    \?)echo "Invalid option: $OPTARG"
      usage; exit 1;;
    esac
done

if [ ! "$preserved_bed_path" ]; then
    echo "Missing option -p"
    usage
    exit 1
fi

if [ ! "$background_bed_path" ]; then
    echo "Missing option -b"
    usage
    exit 1
fi

if [ ! "$tss_path" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

if [ ! "$cleavage_site_path" ]; then
    echo "Missing option -c"
    usage
    exit 1
fi

if [ ! "$genome_path" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi

if [ ! "$id_convert_table_path" ]; then
    echo "Missing option -i"
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
kwargs_path=$saved_root/pipeline_kwargs.csv
echo "name,value" > $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path
echo "upstream_dist,$upstream_dist" >> $kwargs_path
echo "downstream_dist,$downstream_dist" >> $kwargs_path
echo "source_name,$source_name" >> $kwargs_path
echo "redefine_coordinate_kwargs,$redefine_coordinate_kwargs" >> $kwargs_path
echo "filter_kwargs,$filter_kwargs" >> $kwargs_path
echo "process_data_kwargs,$process_data_kwargs" >> $kwargs_path

#Set parameter
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..

echo "Step 1: Redefining coordinate of transcript"
coordinate_redefined_root=$saved_root/coordinate_redefined
coordinate_consist_bed_path=$coordinate_redefined_root/coordinate_consist.bed
if [ ! -e "$coordinate_consist_bed_path" ]; then
    command="$bash_root/redefine_coordinate.sh -b $preserved_bed_path -t $tss_path -c $cleavage_site_path -u $upstream_dist -d $downstream_dist -i $id_convert_table_path -o $coordinate_redefined_root"
    if [ "$redefine_coordinate_kwargs" ]; then
        command="$command $redefine_coordinate_kwargs"
    fi
    echo $command
    eval "bash $command"
fi

background_merged_bed_path=$saved_root/background_and_coordinate_consist.bed
cat $background_bed_path > $background_merged_bed_path
cat $coordinate_consist_bed_path >> $background_merged_bed_path

echo "Step 2: Filtering transcript"
filtered_root=$saved_root/filtered
filtered_bed_path=$filtered_root/filtered.bed
if [ ! -e "$filtered_bed_path" ]; then
    command="$bash_root/filter.sh -i $coordinate_consist_bed_path -t $id_convert_table_path -o $filtered_root"

    if [ "$filter_kwargs" ]; then
        command="$command $filter_kwargs"
    fi
    
    echo $command
    eval "bash $command"
fi

echo "Step 3: Processing data"
processed_root=$saved_root/processed
#if [ ! -e "$processed_root/result/double_strand/canonical_double_strand.bed" ]; then

    command="$bash_root/process_data.sh -p $filtered_bed_path -b $background_merged_bed_path \
    -u $upstream_dist -d $downstream_dist -g $genome_path \
    -t $id_convert_table_path -s $source_name -o $processed_root"
    if [ "$process_data_kwargs" ]; then
        command="$command $process_data_kwargs"
    fi
    echo $command
    eval "bash $command"
#fi

echo "Step 4: Write statistic data of GFF"
result_root=$processed_root/result
double_strand_root=$result_root/double_strand
ds_rna_gff_path=$double_strand_root/rna_double_strand.gff3
ds_canonical_gff_path=$double_strand_root/canonical_double_strand.gff3
ds_region_fasta_path=$double_strand_root/selected_region_double_strand.fasta
region_table_path=$result_root/region_id_conversion.tsv

if [ ! -e "$saved_root/rna_stats/gff_analysis.log" ]; then
    #echo "run rna"
    bash  $bash_root/gff_analysis.sh -i $ds_rna_gff_path -f $ds_region_fasta_path -o $saved_root/rna_stats \
    -r $region_table_path -s ordinal_id_wo_strand
fi

if [ ! -e "$saved_root/canonical_stats/gff_analysis.log" ]; then
    #echo "run gene"
    bash  $bash_root/gff_analysis.sh -i $ds_canonical_gff_path -f $ds_region_fasta_path -o $saved_root/canonical_stats \
    -r $region_table_path -s ordinal_id_wo_strand
fi

echo "Step 5: Splitting data"
splitted_root=$saved_root/split
#if [ ! -e "$splitted_root/double_strand_data/split_without_strand/count.csv" ]; then
    #echo "split ds"
    bash $bash_root/region_select_split.sh -g $genome_path -t $id_convert_table_path -p $processed_root -o $splitted_root -d
#fi

#if [ ! -e "$splitted_root/single_strand_data/split_with_strand/count.csv" ]; then
    echo "split ss"
    bash $bash_root/region_select_split.sh -g $genome_path -t $id_convert_table_path -p $processed_root -o $splitted_root -s
#fi
