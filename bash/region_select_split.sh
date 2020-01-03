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
 echo "    -s  <bool>    Split training and validation dataset with strand"
 echo "    -n  <int>     Fold number (exclude testing dataset), if it is not provided then it would decided by chromosome number"
 echo "    -h            Print help message and exit"
 echo "Example: bash region_select_split.sh -i /home/io/preprocess/id_convert.tsv -o ./data/2019_07_12 -g genome.fasta"
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
split_both_strand_data_root=$saved_root/both_strand_data
split_single_strand_data_root=$saved_root/single_strand_data
mkdir -p $saved_root
mkdir -p $split_both_strand_data_root
mkdir -p $split_single_strand_data_root
    
echo "Start of program"
#Create folder
mkdir -p $processed_root

result_num=$(wc -l < $processed_root/result/selected_region.bed )

if  (( $result_num > 0 )) ; then

    python3 $preprocess_main_root/rename_id_by_coordinate.py -i $processed_root/result/selected_region.bed -p region \
    -t $processed_root/result/region_rename_table_both_strand.tsv -o $processed_root/result/selected_region_both_strand.bed

    python3 $preprocess_main_root/rename_bed_chrom.py -i $processed_root/result/rna.bed \
    -t $processed_root/result/region_rename_table_both_strand.tsv -o $processed_root/result/rna_both_strand.bed
    
    python3 $preprocess_main_root/rename_bed_chrom.py -i $processed_root/result/canonical.bed \
    -t $processed_root/result/region_rename_table_both_strand.tsv -o $processed_root/result/canonical_both_strand.bed
    
    bedtools getfasta -name -fi $genome_path -bed $processed_root/result/selected_region_both_strand.bed \
    -fo $processed_root/result/selected_region_both_strand.fasta


    if ! "$split_with_strand"; then
        command="$preprocess_main_root/split.py --region_rename_table_path $processed_root/result/region_rename_table_both_strand.tsv --fai_path $genome_path.fai --saved_root $split_both_strand_data_root"

        if [ "$fold_num" ]; then
            command="${command} --fold_num $fold_num"
        fi

        if $split_with_strand; then
            command="${command} --split_with_strand"
        fi

        echo $command
        python3 $command

        for path in $(find $split_both_strand_data_root/* -name '*.tsv');
        do
            file_name=$(basename $path)
            file_name="${file_name%.*}"
            python3 $preprocess_main_root/get_subbed.py -i $processed_root/result/rna_both_strand.bed \
            -d $split_both_strand_data_root/$file_name.tsv -o $split_both_strand_data_root/$file_name.bed --query_column chr

            python3 $preprocess_main_root/get_subbed.py -i $processed_root/result/canonical_both_strand.bed \
            -d $split_both_strand_data_root/$file_name.tsv -o $split_both_strand_data_root/${file_name}_canonical.bed --query_column chr

            python3 $preprocess_main_root/bed2gff.py -i $split_both_strand_data_root/$file_name.bed \
            -o $split_both_strand_data_root/$file_name.gff -t $id_convert_table_path

            python3 $preprocess_main_root/bed2gff.py -i $split_both_strand_data_root/${file_name}_canonical.bed \
            -o $split_both_strand_data_root/${file_name}_canonical.gff

            python3 $preprocess_main_root/get_subfasta.py -i $processed_root/result/selected_region_both_strand.fasta \
            -d $split_both_strand_data_root/$file_name.tsv -o $split_both_strand_data_root/$file_name.fasta

            samtools faidx $split_both_strand_data_root/$file_name.fasta
        done
    fi
    
    command="$preprocess_main_root/split.py --region_rename_table_path $processed_root/result/region_rename_table.tsv --fai_path $genome_path.fai --saved_root $split_single_strand_data_root"

    if [ "$fold_num" ]; then
        command="${command} --fold_num $fold_num"
    fi
    
    if $split_with_strand; then
        command="${command} --split_with_strand"
    fi
    
    echo $command
    python3 $command

    for path in $(find $split_single_strand_data_root/* -name '*.tsv');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        python3 $preprocess_main_root/get_subbed.py -i $processed_root/result/rna.bed \
        -d $split_single_strand_data_root/$file_name.tsv -o $split_single_strand_data_root/$file_name.bed --query_column chr
        
        python3 $preprocess_main_root/get_subbed.py -i $processed_root/result/canonical.bed \
        -d $split_single_strand_data_root/$file_name.tsv -o $split_single_strand_data_root/${file_name}_canonical.bed --query_column chr
        
        python3 $preprocess_main_root/get_subfasta.py -i $processed_root/result/selected_region.fasta \
        -d $split_single_strand_data_root/$file_name.tsv -o $split_single_strand_data_root/$file_name.fasta
        
        samtools faidx $split_single_strand_data_root/$file_name.fasta
    done

    exit 0
else
    echo "The program process_data.sh is failed"
    exit 1
fi
