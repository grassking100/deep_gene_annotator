#!/bin/bash
## function print usage
usage(){
 echo "Usage: The pipeline selects regions around transcription"
 echo "  Arguments:"
 echo "    -i  <string>  Path of input gff"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -o  <string>  Directory for output result"
 echo "    -t  <string>  Directory of TransDecoder"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash select_transcription_potential_region.sh -u 1000 -d 1000 -g genome.fasta -i example.bed -o result"
 echo ""
}

while getopts i:u:d:g:o:c:t:h option
 do
  case "${option}"
  in
   i )input_gff_path=$OPTARG;;
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   g )genome_path=$OPTARG;;
   o )saved_root=$OPTARG;;
   c )chroms=$OPTARG;;
   t )transdecoder_root=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$input_gff_path" ]; then
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

if [ ! "$genome_path" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$chroms" ]; then
    echo "Missing option -c"
    usage
    exit 1
fi

if [ ! "$transdecoder_root" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
visual_main_root=$script_root/sequence_annotation/visual
genome_handler_main_root=$script_root/sequence_annotation/genome_handler

#rm -rf $saved_root
mkdir -p $saved_root


kwargs_path=$saved_root/process_kwargs.csv
echo "name,value" > $kwargs_path
echo "input_gff_path,$input_gff_path" >> $kwargs_path
echo "flanking_dist,$flanking_dist" >> $kwargs_path
echo "genome_path,$genome_path" >> $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path

echo "Step 1: Create regions around transcript on double-strand"
all_transcript_gff_path=$saved_root/all_transcript.gff3
coord_bed_path=$saved_root/part_coord.bed
all_coord_bed_path=$saved_root/all_coord.bed
coord_flanking_bed_path=$saved_root/coord_flanking.bed
coord_flanking_fasta_path=$saved_root/coord_flanking.fasta
cleaned_region_bed_path=$saved_root/cleaned_region.bed
merged_region_bed_path=$saved_root/merged_region.bed
potential_region_fasta_path=$saved_root/potential_region.fasta
region_double_strand_fasta_path=$saved_root/region_double_strand.fasta
region_bed_path=$saved_root/region.bed
region_fasta_path=$saved_root/region.fasta
dirty_bed_path=$saved_root/dirty.bed
transcript_id_path=$saved_root/transcript_id.txt
transcript_bed_path=$saved_root/transcript.bed
id_convert_path=$saved_root/id_convert.tsv
alt_region_gff_path=$saved_root/alt_region.gff3
canonical_bed_path=$saved_root/canonical.bed
canonical_gff_path=$saved_root/canonical.gff3

canonical_fasta_path=$saved_root/canonical_cDNA.fasta
input_bed_path=$saved_root/input.bed
region_table_path=$saved_root/region_id_conversion.tsv
coord_region_bed_path=$saved_root/coord_region.bed
canonical_id_table_path=$saved_root/canonical_id_table.tsv

redefined_transcript_bed_path=$saved_root/redefined_transcript.bed
redefined_canonical_bed_path=$saved_root/redefined_canonical.bed

#python3 $preprocess_main_root/get_consistent_gff.py -i $input_gff_path -s $saved_root --postfix 'input' --split_by_parent
#python3 $preprocess_main_root/get_id_table.py -i $input_gff_path -o $id_convert_path

#python3 $preprocess_main_root/get_feature_gff.py -i $saved_root/repaired_input.gff3 -f "RNA" -o $all_transcript_gff_path
#python3 $preprocess_main_root/gff2bed.py -i $all_transcript_gff_path -o $all_coord_bed_path --simple_mode
#python3 $preprocess_main_root/get_subbed.py -i $all_coord_bed_path -o $coord_bed_path --query_column chr -d $chroms --treat_id_path_as_ids
#bedtools slop -s -i $coord_bed_path -g $genome_path.fai -l $upstream_dist -r $downstream_dist > $coord_flanking_bed_path

echo "Step 2: Select region which its sequence only has A, T, C, and G"
#bedtools getfasta -name -fi $genome_path -bed $coord_flanking_bed_path -fo $coord_flanking_fasta_path
#python3 $preprocess_main_root/get_cleaned_code_bed.py -b $coord_flanking_bed_path -f $coord_flanking_fasta_path -o $cleaned_region_bed_path -d $dirty_bed_path

echo "Step 3: Merge region and get transcript id"
#bash $bash_root/sort_merge.sh -i $cleaned_region_bed_path -o $merged_region_bed_path
#awk -F '\t' -v OFS='\t' '{print($1,$2,$3,$4,$5,"+");print($1,$2,$3,$4,$5,"-");}' $merged_region_bed_path > $region_bed_path
#awk -F '\t' -v OFS='\t' '{n=split($4,ids,",");for(i=1;i<=n;i++) printf("%s\n",ids[i])}' $merged_region_bed_path > $transcript_id_path

echo "Step 4: Rename region and get sequence"
#python3 $preprocess_main_root/region_id_conversion.py -i $region_bed_path -p region -t $region_table_path \
#-c $coord_region_bed_path

#bedtools getfasta -s -name -fi $genome_path -bed $coord_region_bed_path -fo $region_fasta_path
#bedtools getfasta -name -fi $genome_path -bed $coord_region_bed_path -fo $region_double_strand_fasta_path
#python3 $preprocess_main_root/rename_chrom.py -i $region_fasta_path -t $region_table_path -o $region_fasta_path --source 'coord' --target 'ordinal_id_with_strand'
#python3 $preprocess_main_root/rename_chrom.py -i $region_double_strand_fasta_path -t $region_table_path -o $region_double_strand_fasta_path --source 'coord' --target 'ordinal_id_wo_strand'

echo "Step 5: Get canonical gene annotation"
#python3 $preprocess_main_root/gff2bed.py -i $saved_root/repaired_input.gff3 -o $input_bed_path
#python3 $preprocess_main_root/get_subbed.py -i $input_bed_path -d $transcript_id_path -o $transcript_bed_path

#python3 $preprocess_main_root/convert_transcript_to_gene_with_alt_status_gff.py -i $transcript_bed_path -t $id_convert_path -o $alt_region_gff_path --select_site_by_election
#python3 $preprocess_main_root/create_gene_bed_from_exon_gff.py -i $alt_region_gff_path -o $canonical_bed_path
#python3 $preprocess_main_root/get_id_table.py -i $alt_region_gff_path -o $canonical_id_table_path
#python3 $preprocess_main_root/bed2gff.py -i $canonical_bed_path -o $canonical_gff_path -t $canonical_id_table_path
#bedtools getfasta -s -name -split -fi $genome_path -bed $canonical_bed_path -fo $canonical_fasta_path

echo "Step 6: Get redefined coordinate bed"
#python3 $preprocess_main_root/redefine_coordinate.py -i $transcript_bed_path -t $region_table_path -o $redefined_transcript_bed_path --chrom_target 'ordinal_id_wo_strand'
#python3 $preprocess_main_root/redefine_coordinate.py -i $canonical_bed_path -t $region_table_path -o $redefined_canonical_bed_path --chrom_target 'ordinal_id_wo_strand'

echo "Step 7: Write statistic data"
num_transcript=$(wc -l < $all_transcript_gff_path )
num_cleaned_region=$(wc -l < $cleaned_region_bed_path )
num_region=$(wc -l < $region_bed_path )
num_transcript=$(wc -l < $transcript_id_path )
num_selected_gene=$(wc -l < $canonical_bed_path )

echo "The number of transcript: $num_transcript" > $saved_root/count.stats
echo "The number of cleaned regions: $num_cleaned_region" >> $saved_root/count.stats
echo "The number of merged regions: $num_region" >> $saved_root/count.stats
echo "The number of transcript in region: $num_transcript" >> $saved_root/count.stats
echo "The number of gene in region: $num_selected_gene" >> $saved_root/count.stats

echo "Step 8: Get peptide"
transdecoder_result_root=$saved_root/transdecoder
mkdir -p $transdecoder_result_root
cd $transdecoder_result_root
perl $transdecoder_root/TransDecoder.LongOrfs -t $canonical_fasta_path > $transdecoder_result_root/long_orf_record.log
perl $transdecoder_root/TransDecoder.Predict -t $canonical_fasta_path > $transdecoder_result_root/predict_record.log
