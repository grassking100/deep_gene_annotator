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
utils_root=$script_root/sequence_annotation/utils
mkdir -p $saved_root

result_num=$(wc -l < $processed_root/result/selected_region.fasta )


if  (( $result_num > 0 )) ; then

    

    command="$preprocess_main_root/split.py --region_path $region_table_path --fai_path $genome_path.fai --saved_root $saved_root"

    if [ "$fold_num" ]; then
        command="${command} --fold_num $fold_num"
    fi

    if $treat_strand_independent && ! $on_double_strand_data ; then
        command="${command} --treat_strand_independent"
    fi

    if $on_double_strand_data ; then
        command="${command} --id_source ordinal_id_wo_strand"
    else
        command="${command} --id_source ordinal_id_with_strand"
    fi

    echo $command
    python3 $command
    
    ds_region_fasta_path=$processed_root/result/double_strand/selected_region_double_strand.fasta
    region_fasta_path=$processed_root/result/selected_region.fasta
    
    if $on_double_strand_data ; then
        output_region_fasta_root=$ds_region_fasta_path
        rna_bed_path=$processed_root/result/double_strand/rna_double_strand.bed
        canonical_bed_path=$processed_root/result/double_strand/canonical_double_strand.bed
    else
        output_region_fasta_root=$region_fasta_path
        rna_bed_path=$processed_root/result/rna.bed
        canonical_bed_path=$processed_root/result/canonical.bed
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
        part_bed_path=$bed_root/$file_name.bed
        part_canonical_bed_path=$bed_root/${file_name}_canonical.bed
        part_gff_path=$gff_root/$file_name.gff3
        part_canonical_gff_path=$gff_root/${file_name}_canonical.gff3
        #
        if ! $on_double_strand_data ; then
            part_ds_bed_path=$bed_root/${file_name}_double_strand.bed
            part_ds_canonical_bed_path=$bed_root/${file_name}_canonical_double_strand.bed
            part_ds_gff_path=$gff_root/${file_name}_double_strand.gff3
            part_ds_canonical_gff_path=$gff_root/${file_name}_canonical_double_strand.gff3
        else
            part_ds_gff_path=$part_gff_path
            part_ds_canonical_gff_path=$part_canonical_gff_path
        fi
        
        python3 $preprocess_main_root/get_subbed.py -i $rna_bed_path -d $source_id_path \
        -o $part_bed_path --query_column chr
        python3 $preprocess_main_root/get_subbed.py -i $canonical_bed_path -d $source_id_path \
        -o $part_canonical_bed_path --query_column chr
        python3 $preprocess_main_root/bed2gff.py -i $part_bed_path -o $part_gff_path -t $id_convert_table_path
        python3 $preprocess_main_root/bed2gff.py -i $part_canonical_bed_path -o $part_canonical_gff_path \
        -t $alt_region_id_table_path
        
        if ! $on_double_strand_data ; then
            python3 $preprocess_main_root/rename_chrom.py -i $part_bed_path -t $region_table_path \
            -o $part_ds_bed_path --source ordinal_id_with_strand --target ordinal_id_wo_strand
            python3 $preprocess_main_root/rename_chrom.py -i $part_canonical_bed_path -t $region_table_path \
            -o $part_ds_canonical_bed_path --source ordinal_id_with_strand --target ordinal_id_wo_strand
            python3 $preprocess_main_root/rename_chrom.py -i $part_gff_path -t $region_table_path \
            -o $part_ds_gff_path --source ordinal_id_with_strand --target ordinal_id_wo_strand
            python3 $preprocess_main_root/rename_chrom.py -i $part_canonical_gff_path -t $region_table_path \
            -o $part_ds_canonical_gff_path --source ordinal_id_with_strand --target ordinal_id_wo_strand
        fi

        python3 $utils_root/get_subfasta.py -i $output_region_fasta_root -d $source_id_path \
        -o $fasta_root/$file_name.fasta
        samtools faidx $fasta_root/$file_name.fasta
        
        gene_num=$(wc -l < $source_id_path ) 
        echo "$file_name,$gene_num" >> $saved_root/count.csv
        part_region_table_path=$region_table_root/${file_name}_part_region_table.tsv
        if ! $on_double_strand_data ; then
            python3 $preprocess_main_root/get_sub_region_table.py -r $region_table_path -i $source_id_path \
            -s ordinal_id_with_strand -o $part_region_table_path
        else
            python3 $preprocess_main_root/get_sub_region_table.py -r $region_table_path -i $source_id_path \
            -s ordinal_id_wo_strand -o $part_region_table_path
        fi
        
        #if [ ! -e "$saved_root/canonical_stats/$file_name/gff_analysis.log" ]; then
        echo "gene"
        #echo $ds_region_fasta_path
        bash $bash_root/gff_analysis.sh -i $part_ds_canonical_gff_path -f $ds_region_fasta_path \
        -o $saved_root/canonical_stats/$file_name -s ordinal_id_wo_strand -r $part_region_table_path
        #fi
        
        #if [ ! -e "$saved_root/rna_stats/$file_name/gff_analysis.log" ]; then
        echo "rna"
        bash $bash_root/gff_analysis.sh -i $part_ds_gff_path -f $ds_region_fasta_path \
        -o $saved_root/rna_stats/$file_name -s ordinal_id_wo_strand -r $part_region_table_path
        #fi
        
        #if [ ! -e "$saved_root/canonical_stats/$file_name/gff_analysis.log" ]; then
        echo "gene"
        #echo $ds_region_fasta_path
        bash $bash_root/gff_analysis.sh -i $part_canonical_gff_path -f $region_fasta_path \
        -o $saved_root/with_strand_canonical_stats/$file_name -s ordinal_id_with_strand -r $part_region_table_path
        #fi
        
        #if [ ! -e "$saved_root/rna_stats/$file_name/gff_analysis.log" ]; then
        echo "rna"
        bash $bash_root/gff_analysis.sh -i $part_gff_path -f $region_fasta_path \
        -o $saved_root/with_strand_rna_stats/$file_name -s ordinal_id_with_strand -r $part_region_table_path
        #fi

    done
fi

exit 0
