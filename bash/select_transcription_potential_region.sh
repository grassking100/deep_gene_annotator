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

while getopts i:u:d:g:o:c:t:m:h option
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
   m )cdhit_matched_id_path=$OPTARG;;
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

if [ ! "$cdhit_matched_id_path" ]; then
    echo "Missing option -m"
    usage
    exit 1
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
visual_main_root=$script_root/sequence_annotation/visual
genome_handler_main_root=$script_root/sequence_annotation/genome_handler

mkdir -p $saved_root

kwargs_path=$saved_root/process_kwargs.csv
echo "name,value" > $kwargs_path
echo "input_gff_path,$input_gff_path" >> $kwargs_path
echo "flanking_dist,$flanking_dist" >> $kwargs_path
echo "genome_path,$genome_path" >> $kwargs_path
echo "chroms,$chroms" >> $kwargs_path
echo "cdhit_matched_id_path,$cdhit_matched_id_path" >> $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path

echo "Step 1: Extract transcript data"
preprocess_root=$saved_root/preprocessed
mkdir -p $preprocess_root
id_convert_path=$preprocess_root/id_convert.tsv
all_transcript_gff_path=$preprocess_root/all_transcript.gff3
repaired_input_gff_path=$preprocess_root/repaired_input.gff3
repaired_input_bed_path=$preprocess_root/repaired_input.bed
#python3 $preprocess_main_root/get_consistent_gff.py -i $input_gff_path -s $saved_root --postfix 'input' --split_by_parent
#python3 $preprocess_main_root/get_id_table.py -i $input_gff_path -o $id_convert_path
#python3 $preprocess_main_root/get_feature_gff.py -i $repaired_input_gff_path -f "RNA" -o $all_transcript_gff_path
#python3 $preprocess_main_root/gff2bed.py -i $repaired_input_gff_path -o $repaired_input_bed_path

echo "Step 2: Create regions around transcript on double-strand"
region_root=$saved_root/region
mkdir -p $region_root
all_coord_bed_path=$region_root/all_coord.bed
coord_bed_path=$region_root/transcript_coord.bed
coord_flanking_bed_path=$region_root/coord_flanking.bed
coord_flanking_fasta_path=$region_root/coord_flanking.fasta
#python3 $preprocess_main_root/gff2bed.py -i $all_transcript_gff_path -o $all_coord_bed_path --simple_mode
#python3 $preprocess_main_root/get_subbed.py -i $all_coord_bed_path -o $coord_bed_path --query_column chr -d $chroms --treat_id_path_as_ids
#bedtools slop -s -i $coord_bed_path -g $genome_path.fai -l $upstream_dist -r $downstream_dist > $coord_flanking_bed_path
#bedtools getfasta -name -fi $genome_path -bed $coord_flanking_bed_path -fo $coord_flanking_fasta_path


echo "Step 3: Select region which its sequence only has A, T, C, and G"
cleaning_root=$saved_root/cleaning
mkdir -p $cleaning_root
cleaned_region_bed_path=$cleaning_root/cleaned_region.bed
dirty_region_bed_path=$cleaning_root/dirty_region.bed
#python3 $preprocess_main_root/get_cleaned_code_bed.py -b $coord_flanking_bed_path -f $coord_flanking_fasta_path -o $cleaned_region_bed_path -d $dirty_region_bed_path

echo "Step 4: Merge region and get transcript id"
merged_root=$saved_root/merged
mkdir -p $merged_root
transcript_id_path=$merged_root/transcript_id.txt
merged_region_bed_path=$merged_root/merged_region.bed
region_single_strand_bed_path=$merged_root/region_single_strand.bed
#bash $bash_root/sort_merge.sh -i $cleaned_region_bed_path -o $merged_region_bed_path
#awk -F '\t' -v OFS='\t' '{print($1,$2,$3,$4,$5,"+");print($1,$2,$3,$4,$5,"-");}' $merged_region_bed_path > $region_single_strand_bed_path
#awk -F '\t' -v OFS='\t' '{n=split($4,idouble_strand,",");for(i=1;i<=n;i++) printf("%s\n",idouble_strand[i])}' $merged_region_bed_path > $transcript_id_path


echo "Step 5: Rename region and get sequence"
result_root=$saved_root/result
mkdir -p $result_root
coord_region_single_strand_bed_path=$result_root/coord_region_single_strand.bed
region_single_strand_fasta_path=$result_root/region_single_strand.fasta
region_double_strand_fasta_path=$result_root/region_double_strand.fasta
region_table_path=$result_root/region_id_conversion.tsv
#python3 $preprocess_main_root/region_id_conversion.py -i $region_single_strand_bed_path -p region -t $region_table_path -c $coord_region_single_strand_bed_path

#bedtools getfasta -s -name -fi $genome_path -bed $coord_region_single_strand_bed_path -fo $region_single_strand_fasta_path
#python3 $preprocess_main_root/rename_chrom.py -i $region_single_strand_fasta_path -t $region_table_path -o $region_single_strand_fasta_path --source 'coord' --target 'ordinal_id_with_strand'

#bedtools getfasta -name -fi $genome_path -bed $coord_region_single_strand_bed_path -fo $region_double_strand_fasta_path
#python3 $preprocess_main_root/rename_chrom.py -i $region_double_strand_fasta_path -t $region_table_path -o $region_double_strand_fasta_path --source 'coord' --target 'ordinal_id_wo_strand'

echo "Step 6: Get canonical gene annotation"
annotation_root=$saved_root/annotation
mkdir -p $annotation_root
transcript_bed_path=$annotation_root/transcript.bed
alt_region_gff_path=$annotation_root/alt_region.gff3
canonical_bed_path=$annotation_root/canonical.bed
canonical_gff_path=$annotation_root/canonical.gff3
canonical_id_table_path=$annotation_root/canonical_id_table.tsv
transcript_fasta_path=$annotation_root/transcript_cDNA.fasta
canonical_fasta_path=$annotation_root/canonical_cDNA.fasta
#python3 $preprocess_main_root/get_subbed.py -i $repaired_input_bed_path -d $transcript_id_path -o $transcript_bed_path
#python3 $preprocess_main_root/create_canonical_gene.py -i $transcript_bed_path  -t $id_convert_path -s $alt_region_gff_path -b $canonical_bed_path -g $canonical_gff_path -o $canonical_id_table_path --select_site_by_election
#bedtools getfasta -s -name -split -fi $genome_path -bed $transcript_bed_path -fo $transcript_fasta_path
#bedtools getfasta -s -name -split -fi $genome_path -bed $canonical_bed_path -fo $canonical_fasta_path

echo "Step 7: Get cd-hit matched data"
cdhit_result_root=$saved_root/cdhit
mkdir -p $cdhit_result_root
cdhit_transcript_bed_path=$cdhit_result_root/cdhit_transcript.bed
cdhit_canonical_bed_path=$cdhit_result_root/cdhit_canonical.bed
cdhit_transcript_fasta_path=$cdhit_result_root/cdhit_transcript_cDNA.fasta
cdhit_canonical_fasta_path=$cdhit_result_root/cdhit_canonical_cDNA.fasta
#python3 $preprocess_main_root/get_subbed.py -i $transcript_bed_path -d $cdhit_matched_id_path -o $cdhit_transcript_bed_path
#python3 $preprocess_main_root/get_subbed.py -i $canonical_bed_path -d $cdhit_matched_id_path -t $id_convert_path -o $cdhit_canonical_bed_path --convert_input_id
#bedtools getfasta -s -name -split -fi $genome_path -bed $cdhit_transcript_bed_path -fo $cdhit_transcript_fasta_path
#bedtools getfasta -s -name -split -fi $genome_path -bed $cdhit_canonical_bed_path -fo $cdhit_canonical_fasta_path
cd $cdhit_result_root
/home/tools/cd-hit-v4.8.1-2019-0228/cd-hit-est-2d -i2 $cdhit_transcript_fasta_path -i $cdhit_canonical_fasta_path -o cdhit_canonical_result -T 0 -M 0 -g 1 -G 1 -s2 0 -c 0.8 -n 5
cd $result_root
python3 $script_root/tools/analysize_cd_hit.py -i $cdhit_result_root/cdhit_canonical_result -o $cdhit_result_root/statistic


echo "Step 8: Get redefined coordinate bed"
redefined_transcript_bed_path=$result_root/transcript_double_strand.bed
redefined_canonical_bed_path=$result_root/canonical_double_strand.bed
#python3 $preprocess_main_root/redefine_coordinate.py -i $transcript_bed_path -t $region_table_path -o $redefined_transcript_bed_path --chrom_target 'ordinal_id_wo_strand'
#python3 $preprocess_main_root/redefine_coordinate.py -i $canonical_bed_path -t $region_table_path -o $redefined_canonical_bed_path --chrom_target 'ordinal_id_wo_strand'

echo "Step 9: Write statistic data"
num_all_transcript=$(wc -l < $coord_bed_path )
num_all_region=$(wc -l < $coord_flanking_bed_path )
num_cleaned_region=$(wc -l < $cleaned_region_bed_path )
num_dirty_region=$(wc -l < $dirty_region_bed_path )
num_region=$(wc -l < $merged_region_bed_path )
num_selected_transcript=$(wc -l < $transcript_id_path )
num_selected_gene=$(wc -l < $canonical_bed_path )

echo "The number of all transcript: $num_all_transcript" > $saved_root/count.stats

echo "The number of all regions: $num_all_region" >> $saved_root/count.stats
echo "The number of cleaned regions: $num_cleaned_region" >> $saved_root/count.stats
echo "The number of dirty regions: $num_dirty_region" >> $saved_root/count.stats
echo "The number of merged regions: $num_region" >> $saved_root/count.stats
echo "The number of transcript in region: $num_selected_transcript" >> $saved_root/count.stats
echo "The number of gene in region: $num_selected_gene" >> $saved_root/count.stats

echo "Step 10: Get peptide"
transdecoder_result_root=$saved_root/transdecoder
mkdir -p $transdecoder_result_root
cd $transdecoder_result_root
#perl $transdecoder_root/TransDecoder.LongOrfs -S -t $canonical_fasta_path > $transdecoder_result_root/long_orf_record.log
#perl $transdecoder_root/TransDecoder.Predict --single_best_only -t $canonical_fasta_path > $transdecoder_result_root/predict_record.log
