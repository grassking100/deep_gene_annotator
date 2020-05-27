#!/bin/bash
# function print usage
usage(){
 echo "Usage: The pipeline selects regions around transcription"
 echo "  Arguments:"
 echo "    -i  <string>  Path of input gff"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -o  <string>  Directory for output result"
 echo "    -c  <string>  Selected chromosomes"
 echo "    -t  <string>  Directory of TransDecoder"
 echo "    -m  <string>  Directory of cd-hit matched id"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash select_transcription_potential_region.sh -u 1000 -d 1000 -g genome.fasta -i example.bed -o result -c Chr1"
 echo ""
}

while getopts i:g:o:c:h option
 do
  case "${option}"
  in
   i )input_gff_path=$OPTARG;;
   g )genome_path=$OPTARG;;
   o )output_root=$OPTARG;;
   c )chroms=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
visual_main_root=$script_root/sequence_annotation/visual
genome_handler_main_root=$script_root/sequence_annotation/genome_handler

mkdir -p $output_root

#kwargs_path=$saved_root/process_kwargs.csv
#echo "name,value" > $kwargs_path
#echo "input_gff_path,$input_gff_path" >> $kwargs_path
#echo "flanking_dist,$flanking_dist" >> $kwargs_path
#echo "genome_path,$genome_path" >> $kwargs_path
#echo "chroms,$chroms" >> $kwargs_path
#echo "cdhit_matched_id_path,$cdhit_matched_id_path" >> $kwargs_path
#echo "saved_root,$saved_root" >> $kwargs_path

bed_path=$output_root/input.bed
part_bed_path=$output_root/part_input.bed
id_convert_path=$output_root/id_convert.tsv
alt_region_gff_path=$output_root/alt_region.gff
canonical_bed_path=$output_root/canonical.bed
canonical_gff_path=$output_root/canonical.gff
canonical_id_table_path=$output_root/canonical_id_table.tsv
cleane_bed_path=$output_root/cleaned.bed
cDNA_path=$output_root/cDNA.fasta

#python3 $preprocess_main_root/get_id_table.py -i $input_gff_path -o $id_convert_path
#python3 $preprocess_main_root/gff2bed.py -i $input_gff_path -o $bed_path
#python3 $preprocess_main_root/get_subbed.py -i $bed_path -o $part_bed_path --query_column chr -d $chroms --treat_id_path_as_ids
python3 $preprocess_main_root/create_canonical_coding_bed.py -i $part_bed_path -t $id_convert_path -o $canonical_gff_path
#python3 $preprocess_main_root/get_cleaned_code_bed.py -b $canonical_bed_path -f $genome_path -o $cleane_bed_path
#bedtools getfasta -name  -s -split -fi $genome_path -bed $cleane_bed_path -fo $cDNA_path
