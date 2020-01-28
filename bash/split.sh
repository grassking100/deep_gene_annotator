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
 echo "Example: bash split.sh -p /home/io/process -t /home/io/preprocess/id_convert.tsv -o ./data/2019_07_12 -g genome.fasta"
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

result_num=$(wc -l < $processed_root/result/selected_region.bed )

if  (( $result_num > 0 )) ; then

    output_region_fasta_path=$processed_root/result/selected_region.fasta
    region_rename_table_path=$processed_root/result/region_rename_table.tsv
    rna_bed_path=$processed_root/result/rna.bed
    canonical_bed_path=$processed_root/result/canonical.bed

    command="$preprocess_main_root/split.py --region_rename_table_path $region_rename_table_path --fai_path $genome_path.fai --saved_root $saved_root"

    if [ "$fold_num" ]; then
        command="${command} --fold_num $fold_num"
    fi

    if $split_with_strand; then
        command="${command} --split_with_strand"
    fi

    echo $command
    python3 $command
    
    echo "Name,Region number" > $saved_root/count.csv
    
    for path in $(find $saved_root/* -name '*.txt');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        python3 $preprocess_main_root/get_subbed.py -i $rna_bed_path -d $saved_root/$file_name.txt -o $saved_root/$file_name.bed --query_column chr

        python3 $preprocess_main_root/get_subbed.py -i $canonical_bed_path -d $saved_root/$file_name.txt -o $saved_root/${file_name}_canonical.bed --query_column chr

        python3 $preprocess_main_root/get_subfasta.py -i $output_region_fasta_path -d $saved_root/$file_name.txt -o $saved_root/$file_name.fasta

        python3 $preprocess_main_root/bed2gff.py -i $saved_root/$file_name.bed -o $saved_root/$file_name.gff -t $id_convert_table_path

        python3 $preprocess_main_root/bed2gff.py -i $saved_root/${file_name}_canonical.bed -o $saved_root/${file_name}_canonical.gff

        samtools faidx $saved_root/$file_name.fasta
        
        gene_num=$(wc -l < $saved_root/$file_name.txt ) 
        echo "$file_name,$gene_num" >> $saved_root/count.csv
        
    done
fi

exit 0
