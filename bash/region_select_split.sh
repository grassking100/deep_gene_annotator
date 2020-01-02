#!/bin/bash
## function print usage
usage(){
 echo "Usage: The pipeline to select and split regions"
 echo "  Arguments:"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -t  <string>  Path of id_convert_table"
 echo "    -o  <string>  Directory of output folder"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "    -n  <int>     Fold number (exclude testing dataset), if it is not provided then it would decided by chromosome number"
 echo "Example: bash region_select_split.sh -i /home/io/preprocess/id_convert.tsv -o ./data/2019_07_12 -g genome.fasta"
 echo ""
}

while getopts g:t:o:n:h option
 do
  case "${option}"
  in
   g )genome_path=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   o )saved_root=$OPTARG;;
   n )fold_num=$OPTARG;;
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

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

#Set parameter
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess

echo "Start of program"
#Create folder
mkdir -p $saved_root

result_num=$(wc -l < $saved_root/result/selected_region.bed )

if  (( $result_num > 0 )) ; then

    python3 $preprocess_main_root/rename_id_by_coordinate.py -i $saved_root/result/selected_region.bed -p region -t $saved_root/result/region_rename_table_both_strand.tsv -o $saved_root/result/selected_region_both_strand.bed

    python3 $preprocess_main_root/rename_bed_chrom.py -i $saved_root/result/rna.bed -t $saved_root/result/region_rename_table_both_strand.tsv -o $saved_root/result/rna_both_strand.bed
    
    python3 $preprocess_main_root/rename_bed_chrom.py -i $saved_root/result/canonical.bed -t $saved_root/result/region_rename_table_both_strand.tsv -o $saved_root/result/canonical_both_strand.bed
    
    bedtools getfasta -name -fi $genome_path -bed $saved_root/result/selected_region_both_strand.bed -fo $saved_root/result/selected_region_both_strand.fasta

    split_root=$saved_root/split
    mkdir -p $split_root

    if [ ! "$fold_num" ]; then
        python3 $preprocess_main_root/split.py --region_rename_table_path $saved_root/result/region_rename_table_both_strand.tsv --fai_path $genome_path.fai --saved_root $split_root
    else
        python3 $preprocess_main_root/split.py --region_rename_table_path $saved_root/result/region_rename_table_both_strand.tsv --fai_path $genome_path.fai --saved_root $split_root --fold_num $fold_num
    fi

    for path in $(find $split_root/* -name '*.tsv');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        python3 $preprocess_main_root/get_subbed.py -i $saved_root/result/rna_both_strand.bed -d $split_root/$file_name.tsv \
        -o $split_root/$file_name.bed --query_column chr
        python3 $preprocess_main_root/get_subbed.py -i $saved_root/result/canonical_both_strand.bed -d $split_root/$file_name.tsv \
        -o $split_root/${file_name}_canonical.bed --query_column chr
        python3 $preprocess_main_root/get_GlimmerHMM_cds_file.py -i $split_root/$file_name.bed -o $split_root/$file_name.cds
        python3 $preprocess_main_root/bed2gff.py -i $split_root/$file_name.bed -o $split_root/$file_name.gff -t $id_convert_table_path
        python3 $preprocess_main_root/bed2gff.py -i $split_root/${file_name}_canonical.bed -o $split_root/${file_name}_canonical.gff
        python3 $preprocess_main_root/get_subfasta.py -i $saved_root/result/selected_region_both_strand.fasta -d $split_root/$file_name.tsv -o $split_root/$file_name.fasta
        samtools faidx $split_root/$file_name.fasta
    done
    
    single_strand_split_root=$saved_root/single_strand_split
    mkdir -p $single_strand_split_root

    if [ ! "$fold_num" ]; then
        python3 $preprocess_main_root/split.py --region_rename_table_path $saved_root/result/region_rename_table.tsv --fai_path $genome_path.fai --saved_root $single_strand_split_root
    else
        python3 $preprocess_main_root/split.py --region_rename_table_path $saved_root/result/region_rename_table.tsv --fai_path $genome_path.fai --saved_root $single_strand_split_root --fold_num $fold_num
    fi

    for path in $(find $single_strand_split_root/* -name '*.tsv');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        python3 $preprocess_main_root/get_subbed.py -i $saved_root/result/rna.bed -d $single_strand_split_root/$file_name.tsv \
        -o $single_strand_split_root/$file_name.bed --query_column chr
        
        python3 $preprocess_main_root/get_subbed.py -i $saved_root/result/canonical.bed -d $single_strand_split_root/$file_name.tsv \
        -o $single_strand_split_root/${file_name}_canonical.bed --query_column chr
        
        python3 $preprocess_main_root/get_subfasta.py -i $saved_root/result/selected_region.fasta -d $single_strand_split_root/$file_name.tsv -o $single_strand_split_root/$file_name.fasta
        samtools faidx $single_strand_split_root/$file_name.fasta
    done

    exit 0
else
    echo "The program process_data.sh is failed"
    exit 1
fi
