#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline select and split regions"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -r  <string>  Directory of Arabidopsis thaliana data"
 echo "    -o  <string>  Directory of preprocess folder"
 echo "    -o  <string>  Directory of output folder"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -m  <bool>    Merge regions which are overlapped                 [default: false]"
 echo "    -c  <bool>    Remove gene with altenative donor site and accept site    [default: false]"
 echo "    -x  <bool>    Remove gene with non-coding transcript    [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash region_select_split.sh -u 10000 -d 10000 -p /home/io/preprocess -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:r:p:o:s:mcxh option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   r )root=$OPTARG;;
   p )preprocessed_root=$OPTARG;;
   o )saved_root=$OPTARG;;
   s )source_name=$OPTARG;;
   m )merge_overlapped=true;;
   c )remove_alt_site=true;;
   x )remove_non_coding=true;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$root" ]; then
    echo "Missing option -r"
    usage
    exit 1
fi

if [ ! "$preprocessed_root" ]; then
    echo "Missing option -p"
    usage
    exit 1
fi

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

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess

echo "Start of program"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root

#Set parameter
genome_path=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
id_convert_table_path=$preprocessed_root/id_convert.tsv
processed_bed_path=$preprocessed_root/processed.bed

if [ ! -e "$saved_root/result/selected_region.bed" ]; then
    command="$bash_root/process_data.sh -u $upstream_dist -d $downstream_dist -g $genome_path -i $preprocessed_root/coordinate_consist.bed -o $saved_root -s $source_name -t $id_convert_table_path -b $processed_bed_path"

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

result_num=$(wc -l < $saved_root/result/selected_region.bed )

if  (( $result_num > 0 )) ; then

    python3 $preprocess_main_root/rename_id_by_coordinate.py -i $saved_root/result/selected_region.bed -p region -t $saved_root/result/region_rename_table_both_strand.tsv -o $saved_root/result/selected_region_both_strand.bed

    python3 $preprocess_main_root/rename_bed_chrom.py -i $saved_root/result/rna.bed -t $saved_root/result/region_rename_table_both_strand.tsv -o $saved_root/result/rna_both_strand.bed
    
    python3 $preprocess_main_root/rename_bed_chrom.py -i $saved_root/result/canonical.bed -t $saved_root/result/region_rename_table_both_strand.tsv -o $saved_root/result/canonical_both_strand.bed
    
    bedtools getfasta -name -fi $genome_path -bed $saved_root/result/selected_region_both_strand.bed -fo $saved_root/result/selected_region_both_strand.fasta

    split_root=$saved_root/split
    mkdir -p $split_root

    python3 $preprocess_main_root/split.py --region_bed_path $saved_root/result/selected_region_both_strand.bed --region_rename_table_path $saved_root/result/region_rename_table_both_strand.tsv --fai_path $genome_path.fai --splitted_id_root $split_root

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

    python3 $preprocess_main_root/split.py --region_bed_path $saved_root/result/selected_region.bed --region_rename_table_path $saved_root/result/region_rename_table.tsv --fai_path $genome_path.fai --splitted_id_root $single_strand_split_root

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
