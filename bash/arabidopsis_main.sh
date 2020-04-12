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
 echo "    -m  <bool>    Merge regions which are overlapped                                                       [default: false]"
 echo "    -c  <bool>    Remove gene with altenative donor site and acceptor site                                 [default: false]"
 echo "    -i  <bool>    Remove regions which inner signal                                                        [default: false]"
 echo "    -x  <bool>    Remove gene with non-coding transcript                                                   [default: false]"
 echo "    -n  <bool>    Remove hypothetical protein                                                              [default: false]"
 echo "    -z  <str>     Mode to select for comparing score, valid options are 'bigger_or_equal', "
 echo "                  'smaller_or_equal'                                                                       [default:bigger_or_equal]"
 echo "    -f  <float>   BED item to preserved when comparing score and threshold, defualt would ignore score"
 echo "    -y  <float>   If it is true, then the gene with transcript which has failed to passed the score "
 echo "                  filter would be removed. Otherwise, only the transcript which has failed to passed the "
 echo "                  score filter would be removed                                                            [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_main.sh -u 10000 -d 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:r:o:s:f:z:mcxiynh option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   r )root=$OPTARG;;
   o )saved_root=$OPTARG;;
   s )source_name=$OPTARG;;
   f )score_filter=$OPTARG;;
   z )compared_mode=$OPTARG;;
   m )merge_overlapped=true;;
   c )remove_alt_site=true;;
   x )remove_non_coding=true;;
   i )remove_inner_end=true;;
   y )remove_fail_score_gene=true;;
   n )remove_non_hypothetical_gene=true;;
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

if [ ! "$remove_fail_score_gene" ]; then
    remove_fail_score_gene=false
fi

if [ ! "$compared_mode" ]; then
    compared_mode=bigger_or_equal
fi

if [ ! "$remove_non_hypothetical_gene" ]; then
    remove_non_hypothetical_gene=false
fi

#Set parameter
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocessed_root=$saved_root/preprocessed
processed_root=$saved_root/processed
result_root=$processed_root/result
splitted_root=$saved_root/split
id_convert_table_path=$preprocessed_root/id_convert.tsv
processed_bed_path=$preprocessed_root/processed.bed
preprocess_main_root=$script_root/sequence_annotation/preprocess
genome_path=$preprocessed_root/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
ds_rna_bed_path=$processed_root/double_strand/rna_double_strand.bed
ds_region_fasta_path=$processed_root/double_strand/selected_region_double_strand.fasta
region_rename_table_double_strand_path=$processed_root/result/double_strand/region_rename_table_double_strand.tsv

mkdir -p $saved_root

echo "name,value" > $saved_root/main_kwargs.csv
echo "root,$root" >> $saved_root/main_kwargs.csv
echo "saved_root,$saved_root" >> $saved_root/main_kwargs.csv
echo "upstream_dist,$upstream_dist" >> $saved_root/main_kwargs.csv
echo "downstream_dist,$downstream_dist" >> $saved_root/main_kwargs.csv
echo "source_name,$source_name" >> $saved_root/main_kwargs.csv
echo "score_filter,$score_filter" >> $saved_root/main_kwargs.csv
echo "compared_mode,$compared_mode" >> $saved_root/main_kwargs.csv
echo "merge_overlapped,$merge_overlapped" >> $saved_root/main_kwargs.csv
echo "remove_alt_site,$remove_alt_site" >> $saved_root/main_kwargs.csv
echo "remove_non_coding,$remove_non_coding" >> $saved_root/main_kwargs.csv
echo "remove_inner_end,$remove_inner_end" >> $saved_root/main_kwargs.csv
echo "remove_fail_score_gene,$remove_fail_score_gene" >> $saved_root/main_kwargs.csv
echo "remove_non_hypothetical_gene,$remove_non_hypothetical_gene" >> $saved_root/main_kwargs.csv

command="$bash_root/arabidopsis_data_prepair.sh -u $upstream_dist -d $downstream_dist -r $root -o $preprocessed_root -s $source_name"
if $remove_inner_end; then
    command="$command -i"
fi
echo $command
bash $command

if [ ! -e "$processed_root/result/double_strand/canonical_double_strand.bed" ]; then
    command="$bash_root/process_data.sh -u $upstream_dist -d $downstream_dist -g $genome_path -i $preprocessed_root/coordinate_consist.bed -o $processed_root -s $source_name -t $id_convert_table_path -b $processed_bed_path"

    if $remove_fail_score_gene; then
        command="$command -y"
    fi

    if $merge_overlapped; then
        command="$command -m"
    fi
    
    if $remove_alt_site; then
        command="$command -c"
    fi
    
    if $remove_non_coding; then
        command="$command -x"
    fi
    
    if $remove_non_hypothetical_gene; then
        command="$command -l $preprocessed_root/non_hypothetical_gene_id.txt"
    fi

    if [ "$score_filter" ]; then
        command="$command -f $score_filter -z $compared_mode"
    fi
    
    echo $command
    bash $command
else
    echo "The program process_data.sh is skipped"
fi


echo "Splitting data"
if [ ! -e "$plitted_root/double_strand_data/split_without_strand/count.csv" ]; then
    bash $bash_root/region_select_split.sh -g $genome_path -r $region_rename_table_double_strand_path -t $id_convert_table_path -p $processed_root -o $splitted_root -d

fi

if [ ! -e "$plitted_root/single_strand_data/split_with_strand/count.csv" ]; then
    bash $bash_root/region_select_split.sh -g $genome_path -r $region_rename_table_double_strand_path -t $id_convert_table_path -p $processed_root -o $splitted_root -s
fi
