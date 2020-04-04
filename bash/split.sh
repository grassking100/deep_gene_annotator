#!/bin/bash
## function print usage
usage(){
 echo "Usage: The pipeline to select and split regions"
 echo "  Arguments:"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -t  <string>  Path of id_convert_table"
 echo "    -r  <string>  Path of region_table"
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

while getopts g:t:r:p:o:n:sdh option
 do
  case "${option}"
  in
   g )genome_path=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   r )region_table_path=$OPTARG;;
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

if [ ! "$region_table_path" ]; then
    echo "Missing option -r"
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

    region_path=$processed_root/result/double_strand/region_rename_table_double_strand.tsv

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
    
    df_region_fasta_path=$processed_root/result/double_strand/selected_region_double_strand.fasta
    region_fasta_path=$processed_root/result/selected_region.fasta
    
    if $on_double_strand_data ; then
        output_region_fasta_root=$df_region_fasta_path
        rna_bed_root=$processed_root/result/double_strand/rna_double_strand.bed
        canonical_bed_root=$processed_root/result/double_strand/canonical_double_strand.bed
    else
        output_region_fasta_root=$region_fasta_path
        rna_bed_root=$processed_root/result/rna.bed
        canonical_bed_root=$processed_root/result/canonical.bed
    fi
    alt_region_id_table_path=$processed_root/result/alt_region_id_table.tsv
    bed_root=$saved_root/bed
    fasta_root=$saved_root/fasta
    gff_root=$saved_root/gff
    length_gaussian_root=$saved_root/length_gaussian
    region_table_root=$saved_root/region_table

    mkdir -p $bed_root
    mkdir -p $fasta_root
    mkdir -p $gff_root
    mkdir -p $region_table_root
    
    echo "Name,Region number" > $saved_root/count.csv
    
    for path in $(find $saved_root/* -name '*.txt');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        source_id_path=$saved_root/$file_name.txt
        #
        single_strand_bed=$bed_root/$file_name.bed
        single_strand_canonical_bed=$bed_root/${file_name}_canonical.bed
        single_strand_gff=$gff_root/$file_name.gff3
        single_strand_canonical_gff=$gff_root/${file_name}_canonical.gff3
        #
        double_strand_bed=$bed_root/${file_name}_double_strand.bed
        double_strand_canonical_bed=$bed_root/${file_name}_canonical_double_strand.bed
        double_strand_gff=$gff_root/${file_name}_double_strand.gff3
        double_strand_canonical_gff=$gff_root/${file_name}_canonical_double_strand.gff3
        
        region_table_double_strand=$region_table_root/${file_name}_region_table_double_strand.tsv
        #
        python3 $preprocess_main_root/get_subbed.py -i $rna_bed_root -d $source_id_path -o $single_strand_bed --query_column chr
        python3 $preprocess_main_root/get_subbed.py -i $canonical_bed_root -d $source_id_path -o $single_strand_canonical_bed --query_column chr
        #
        python3 $preprocess_main_root/bed2gff.py -i $single_strand_bed -o $single_strand_gff -t $id_convert_table_path
        python3 $preprocess_main_root/bed2gff.py -i $single_strand_canonical_bed -o $single_strand_canonical_gff  -t $alt_region_id_table_path
        
        #if ! $on_double_strand_data ; then
        python3 $preprocess_main_root/rename_chrom.py -i $single_strand_bed -t $region_table_path -o $double_strand_bed
        python3 $preprocess_main_root/rename_chrom.py -i $single_strand_canonical_bed -t $region_table_path -o $double_strand_canonical_bed
        python3 $preprocess_main_root/rename_chrom.py -i $single_strand_gff -t $region_table_path -o $double_strand_gff
        python3 $preprocess_main_root/rename_chrom.py -i $single_strand_canonical_gff -t $region_table_path -o $double_strand_canonical_gff
        #fi

        python3 $preprocess_main_root/get_subfasta.py -i $output_region_fasta_root -d $source_id_path -o $fasta_root/$file_name.fasta
        samtools faidx $fasta_root/$file_name.fasta
        
        gene_num=$(wc -l < $source_id_path ) 
        echo "$file_name,$gene_num" >> $saved_root/count.csv
        
        if ! $on_double_strand_data ; then
            python3 $preprocess_main_root/get_sub_region_table.py -r $processed_root/result/double_strand/region_rename_table_double_strand.tsv -i $source_id_path -s new_id -o $region_table_double_strand
        else
            python3 $preprocess_main_root/get_sub_region_table.py -r $processed_root/result/double_strand/region_rename_table_double_strand.tsv -i $source_id_path -s old_id -o $region_table_double_strand
        fi
        
        if [ ! -e "$saved_root/canonical_stats/$file_name/gff_analysis.log" ]; then
            bash $bash_root/gff_analysis.sh -i $double_strand_canonical_gff -f $df_region_fasta_path -o $saved_root/canonical_stats/$file_name -s new_id -r $region_table_double_strand
        fi
        
        if [ ! -e "$saved_root/rna_stats/$file_name/gff_analysis.log" ]; then
            bash $bash_root/gff_analysis.sh -i $double_strand_gff -f $df_region_fasta_path -o $saved_root/rna_stats/$file_name -s new_id -r $region_table_double_strand
        fi

    done
fi

exit 0
