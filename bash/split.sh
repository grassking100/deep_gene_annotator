#!/bin/bash
## function print usage
usage(){
 echo "Usage: The pipeline to select and split regions"
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
 echo "Example: bash split.sh -p /home/io/process -t /home/io/preprocess/id_convert.tsv -o ./data/2019_07_12 -g genome.fasta"
 echo ""
}

while getopts g:t:p:o:n:sdh option
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

if [ ! "$treat_strand_independent" ]; then
    treat_strand_independent=false
fi

if [ ! "$on_double_strand_data" ]; then
    on_double_strand_data=false
fi

#Set parameter
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
mkdir -p $saved_root

result_num=$(wc -l < $processed_root/result/selected_region.bed )

if  (( $result_num > 0 )) ; then

    region_path=$processed_root/result/region_rename_table_both_strand.tsv

    command="$preprocess_main_root/split.py --region_path $region_path --fai_path $genome_path.fai --saved_root $saved_root"

    if [ "$fold_num" ]; then
        command="${command} --fold_num $fold_num"
    fi

    if $treat_strand_independent && ! $on_double_strand_data ; then
        command="${command} --treat_strand_independent"
    fi

    if $on_double_strand_data ; then
        command="${command} --id_source new_id"
    else
        command="${command} --id_source old_id"
    fi

    echo $command
    python3 $command
    
    echo "Name,Region number" > $saved_root/count.csv
    
    if $on_double_strand_data ; then
        output_region_fasta_path=$processed_root/result/selected_region_both_strand.fasta
        rna_bed_path=$processed_root/result/rna_both_strand.bed
        canonical_bed_path=$processed_root/result/canonical_both_strand.bed
    else
        output_region_fasta_path=$processed_root/result/selected_region.fasta
        rna_bed_path=$processed_root/result/rna.bed
        canonical_bed_path=$processed_root/result/canonical.bed
    fi
    alt_region_id_table_path=$processed_root/result/alt_region_id_table.tsv
    bed_path=$saved_root/bed
    fasta_path=$saved_root/fasta
    gff_path=$saved_root/gff
    stats_path=$saved_root/stats
    
    mkdir -p $bed_path
    mkdir -p $fasta_path
    mkdir -p $gff_path
    mkdir -p $stats_path
    
    for path in $(find $saved_root/* -name '*.txt');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        source_id_path=$saved_root/$file_name.txt
        
        python3 $preprocess_main_root/get_subbed.py -i $rna_bed_path -d $source_id_path -o $bed_path/$file_name.bed --query_column chr

        python3 $preprocess_main_root/get_subbed.py -i $canonical_bed_path -d $source_id_path -o $bed_path/${file_name}_canonical.bed --query_column chr

        python3 $preprocess_main_root/get_subfasta.py -i $output_region_fasta_path -d $source_id_path -o $fasta_path/$file_name.fasta

        python3 $preprocess_main_root/bed2gff.py -i $bed_path/$file_name.bed -o $gff_path/$file_name.gff -t $id_convert_table_path

        python3 $preprocess_main_root/bed2gff.py -i $bed_path/${file_name}_canonical.bed -o $gff_path/${file_name}_canonical.gff  -t $alt_region_id_table_path

        samtools faidx $fasta_path/$file_name.fasta
        
        gene_num=$(wc -l < $source_id_path ) 
        echo "$file_name,$gene_num" >> $stats_path/count.csv
        
    done
fi

exit 0
