#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating annotation data"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -i  <string>  Path of bed"
 echo "    -o  <string>  Directory for output"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -m  <bool>    Merge regions which are overlapped                          [default: false]"
 echo "    -c  <bool>    Remove gene with altenative donor site and acceptor site    [default: false]"
 echo "    -x  <bool>    Remove gene with non-coding transcript                      [default: false]"
 echo "    -t  <string>  Gene and mRNA id converted table path, it will be created if it doesn't be provided"
 echo "    -b  <string>  Path of background bed, it will be set by input path if it doesn't be provided"
 echo "    -z  <str>     Mode to select for comparing score, valid options are 'bigger_or_equal', 'smaller_or_equal' [default:bigger_or_equal]"
 echo "    -f  <float>   BED item to preserved when comparing score and threshold, defualt would ignore score"
 echo "    -y  <float>   If it is true, then the gene with transcript which has failed to passed the score filter would be removed. Otherwise, only the transcript which has failed to passed the score filter would be removed [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash process_data.sh -u 10000 -d 10000 -g /home/io/genome.fasta -i /home/io/example.bed -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:g:i:o:s:t:b:f:z:mcxyh option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   g )genome_path=$OPTARG;;
   i )bed_path=$OPTARG;;
   o )saved_root=$OPTARG;;
   s )source_name=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   b )background_bed_path=$OPTARG;;
   f )score_filter=$OPTARG;;
   z )compared_mode=$OPTARG;;
   m )merge_overlapped=true;;
   c )remove_alt_site=true;;
   x )remove_non_coding=true;;
   y )remove_fail_score_gene=true;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

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

if [ ! "$genome_path" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi

if [ ! "$bed_path" ]; then
    echo "Missing option -i"
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

if [ ! "$background_bed_path" ]; then
    echo "Use $bed_path as background_bed_path"
    background_bed_path=$bed_path
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

if [ ! "$remove_fail_score_gene" ]; then
    remove_fail_score_gene=false
fi

if [ ! "$compared_mode" ]; then
    compared_mode=bigger_or_equal
fi

#root
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
result_root=$saved_root/result
cleaning_root=$saved_root/cleaning
fasta_root=$saved_root/fasta
region_selected_root=$saved_root/region_selected
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
genome_handler_root=$script_root/sequence_annotation/genome_handler
#variable
genome_fai=$genome_path.fai
selected_rna_bed_path=$region_selected_root/rna.bed
selected_gene_bed_path=$region_selected_root/gene.bed
region_bed_path=$result_root/selected_region.bed
rna_bed_path=$result_root/rna.bed
rna_gff_path=$result_root/rna.gff3
canonical_bed_path=$result_root/canonical.bed
region_table_path=$result_root/region_rename_table.tsv
canonical_gff_path=$result_root/canonical.gff3
alt_region_gff_path=$result_root/alt_region.gff3
canonical_h5_path=$result_root/canonical.h5
alt_region_h5_path=$result_root/alt_region.h5
alt_region_id_table_path=$result_root/alt_region_id_table.tsv
double_strand_root=$result_root/double_strand

echo "Start process_main.sh"
#Create folder
echo "Create folder and create id table"

rm -rf $saved_root

mkdir -p $saved_root
mkdir -p $result_root
mkdir -p $cleaning_root
mkdir -p $region_selected_root
mkdir -p $fasta_root
mkdir -p $double_strand_root

cp $bed_path $saved_root/input.bed


if [ ! "$id_convert_table_path" ]; then
    id_convert_table_path=$saved_root/id_table.tsv
    python3 $preprocess_main_root/create_id_table_by_coord.py -i $saved_root/input.bed -o $id_convert_table_path -p gene
    
fi

echo "Step 1: Remove gene or transcript which doesn't pass threshold filter"

if [ "$score_filter" ]; then
    command="$preprocess_main_root/bed_score_filter.py -i $saved_root/input.bed -o $cleaning_root/pass_score_filter.bed --threshold $score_filter --mode $compared_mode -t $id_convert_table_path"
    if $remove_fail_score_gene ; then
        command="$command --remove_gene"
    fi
    echo "python3 $command"
    python3 $command
else
    cp $saved_root/input.bed $cleaning_root/pass_score_filter.bed
fi

echo "Step 2: Remove overlapped gene on the same strand"
python3 $preprocess_main_root/nonoverlap_filter.py -i $cleaning_root/pass_score_filter.bed -s $cleaning_root --use_strand -t $id_convert_table_path 

echo "Step 3 (Optional): Remove transcript which its gene has alternative donor or acceptor site"
if $remove_alt_site ; then
    python3 $preprocess_main_root/remove_alt_site.py -i $cleaning_root/nonoverlap.bed \
    -o $cleaning_root/pass_no_alt_site_filter.bed -t $id_convert_table_path
else
    cp $cleaning_root/nonoverlap.bed $cleaning_root/pass_no_alt_site_filter.bed
fi

echo "Step 4 (Optional): Remove transcript which its gene has non-coding transcript"
if $remove_non_coding ; then
    python3 $preprocess_main_root/belonging_gene_coding_filter.py -i $cleaning_root/pass_no_alt_site_filter.bed -t $id_convert_table_path -o $cleaning_root/pass_no_non_coding_filter.bed
else
    cp $cleaning_root/pass_no_alt_site_filter.bed $cleaning_root/pass_no_non_coding_filter.bed
fi

echo "Step 5: Remove wanted RNAs which are overlapped with unwanted data in specific distance on both strands"
python3 $preprocess_main_root/recurrent_cleaner.py -r $background_bed_path -c $cleaning_root/pass_no_non_coding_filter.bed -f $genome_fai \
-u $upstream_dist -d $downstream_dist -s $cleaning_root -t $id_convert_table_path > $cleaning_root/recurrent.log

echo "Step 6: Create regions around wanted RNAs on both strand"
cp $cleaning_root/recurrent_cleaned.bed $selected_rna_bed_path

#Get potential regions

awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,$4,$5,$6)
}' $selected_rna_bed_path > $selected_gene_bed_path

python3 $preprocess_main_root/convert_bed_id.py -i $selected_gene_bed_path -t $id_convert_table_path -o $selected_gene_bed_path
python3 $preprocess_main_root/remove_coordinate_duplicated_bed.py -i $selected_gene_bed_path -o $selected_gene_bed_path
bedtools slop -s -i $selected_gene_bed_path -g $genome_fai -l $upstream_dist -r $downstream_dist > $region_selected_root/potential_region.bed

awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,".",".","+")
    print($1,$2,$3,".",".","-")
}' $region_selected_root/potential_region.bed > $region_selected_root/potential_region.temp \
&& mv $region_selected_root/potential_region.temp $region_selected_root/potential_region.bed

bash $bash_root/gene_to_region_mapping.sh -i $region_selected_root/potential_region.bed -d $selected_gene_bed_path \
-o $region_selected_root/rna_region_mapping.bed -c $region_selected_root/rna_region_mapping.count

if  $merge_overlapped ; then
    echo "Merge region on same strand"
    bash $bash_root/sort_merge.sh -i $region_selected_root/rna_region_mapping.bed -o $region_selected_root/selected_region.bed -s
else
    echo "Filter region with region_gene_count_filter"
    python3 $preprocess_main_root/region_gene_count_filter.py -i $region_selected_root/rna_region_mapping.bed -r $selected_gene_bed_path -o $region_selected_root/selected_region.bed -e $region_selected_root/discarded_region.bed -u $upstream_dist -d $downstream_dist
fi

echo "Step 7: Select region which its sequence only has A, T, C, and G"
bedtools getfasta -name -fi $genome_path -bed $region_selected_root/selected_region.bed -fo $region_selected_root/temp.fasta
python3 $preprocess_main_root/get_cleaned_code_bed.py -b $region_selected_root/selected_region.bed -f $region_selected_root/temp.fasta -o $region_selected_root/cleaned_selected_region.bed -d $region_selected_root/dirty_selected_region.bed
rm $region_selected_root/temp.fasta
cp $region_selected_root/cleaned_selected_region.bed $region_bed_path

echo "Step 8: Get RNAs in selected regions"
python3 $preprocess_main_root/get_id.py -i $region_bed_path -o $result_root/gene.id
python3 $preprocess_main_root/get_subbed.py -i $selected_rna_bed_path -d $result_root/gene.id \
-o $rna_bed_path -t $id_convert_table_path
python3 $preprocess_main_root/bed2gff.py -i $rna_bed_path -o $rna_gff_path -t $id_convert_table_path

echo "Step 9: Rename region and get fasta"
echo "Caution: The dataset's strand which is single-strand data is indicated its location of origin's strand not new strand!!!"
python3 $preprocess_main_root/rename_id_by_coordinate.py -i $region_bed_path -p region -t $region_table_path \
-o $region_bed_path --use_strand --coord_id_as_old_id --ignore_output_strand

bedtools getfasta -s -name -fi $genome_path -bed $region_bed_path -fo $result_root/selected_region.fasta


if [ -e "$result_root/selected_region.fasta.fai" ]; then
    rm $result_root/selected_region.fasta.fai
fi

samtools faidx $result_root/selected_region.fasta

cp $rna_bed_path $result_root/origin_rna.bed

python3 $preprocess_main_root/redefine_coordinate.py -i $rna_bed_path -t $region_table_path -o $rna_bed_path

echo "Step 10: Create gff about gene structure and alternative status"
python3 $preprocess_main_root/convert_transcript_to_gene_with_alt_status_gff.py -i $rna_bed_path -o $alt_region_gff_path -t $id_convert_table_path

python3 $preprocess_main_root/create_gene_bed_from_exon_gff.py -i $alt_region_gff_path -o $canonical_bed_path

python3 $preprocess_main_root/get_id_table.py -i $alt_region_gff_path -o $alt_region_id_table_path
python3 $preprocess_main_root/bed2gff.py -i $canonical_bed_path -o $canonical_gff_path -t $alt_region_id_table_path

if $remove_alt_site ; then
    python3 $preprocess_main_root/create_ann_genome.py -i $canonical_gff_path -r $region_bed_path -o $canonical_h5_path -s source_name --discard_alt_region --discard_UTR_CDS
    
else
    python3 $preprocess_main_root/create_ann_genome.py -i $canonical_gff_path -r $region_bed_path -o $canonical_h5_path -s source_name  --discard_UTR_CDS
fi

python3 $preprocess_main_root/create_ann_genome.py -i $alt_region_gff_path \
    -r $region_bed_path -o $alt_region_h5_path -s source_name --with_alt_region --with_alt_site_region

echo "Step 11: Write statistic data"
num_input_RNAs=$(wc -l < $bed_path )
num_background_RNAs=$(wc -l < $background_bed_path )
num_pass_score_filter=$(wc -l < $cleaning_root//pass_score_filter.bed )
num_nonoverlap=$(wc -l < $cleaning_root/nonoverlap.bed )
num_pass_no_alt_site_filter=$(wc -l < $cleaning_root/pass_no_alt_site_filter.bed )
num_pass_no_non_coding_filter=$(wc -l < $cleaning_root/pass_no_non_coding_filter.bed )
num_recurrent=$(wc -l < $cleaning_root/recurrent_cleaned.bed )
num_cleaned_region=$(wc -l < $region_selected_root/selected_region.bed )
num_dirty_region=$(wc -l < $region_selected_root/dirty_selected_region.bed )
num_final_rna=$(wc -l < $rna_bed_path )
num_final_region=$(wc -l < $region_bed_path )
num_canonical=$(wc -l < $canonical_bed_path )

echo "The number of background RNAs: $num_background_RNAs" > $saved_root/count.stats
echo "The number of selected to input RNAs: $num_input_RNAs" >> $saved_root/count.stats
echo "The number of RNAs which pass score filter: $num_pass_score_filter" >> $saved_root/count.stats
echo "The number of RNAs which are not overlap with each other: $num_nonoverlap" >> $saved_root/count.stats
echo "The number of RNAs which their gene have passed no-alt-site filter: $num_pass_no_alt_site_filter" >> $saved_root/count.stats
echo "The number of RNAs which their gene have passed no-non-coding filter: $num_pass_no_non_coding_filter" >> $saved_root/count.stats
echo "The number of recurrent-cleaned RNAs: $num_recurrent" >> $saved_root/count.stats
echo "The number of cleaned regions: $num_cleaned_region" >> $saved_root/count.stats
echo "The number of dirty regions: $num_dirty_region" >> $saved_root/count.stats
echo "The number of final regions: $num_final_region" >> $saved_root/count.stats
echo "The number of cleaned genes in valid regions: $num_canonical" >> $saved_root/count.stats
echo "The number of cleaned RNAs in valid regions: $num_final_rna" >> $saved_root/count.stats

echo "Step 12: Create double-strand data"
ds_region_table_path=$double_strand_root/region_rename_table_double_strand.tsv
ds_region_bed_path=$double_strand_root/selected_region_double_strand.bed
ds_region_fasta_path=$double_strand_root/selected_region_double_strand.fasta
ds_rna_bed_path=$double_strand_root/rna_double_strand.bed
ds_rna_gff_path=$double_strand_root/rna_double_strand.gff3
ds_canonical_bed_path=$double_strand_root/canonical_double_strand.bed
ds_canonical_gff_path=$double_strand_root/canonical_double_strand.gff3

python3 $preprocess_main_root/rename_id_by_coordinate.py -i $region_bed_path -p region -t $ds_region_table_path -o $ds_region_bed_path
bedtools getfasta -name -fi $genome_path -bed $ds_region_bed_path -fo $ds_region_fasta_path
python3 $preprocess_main_root/rename_chrom.py -i $rna_bed_path -t $ds_region_table_path -o $ds_rna_bed_path
python3 $preprocess_main_root/rename_chrom.py -i $canonical_bed_path -t $ds_region_table_path -o $ds_canonical_bed_path
python3 $preprocess_main_root/bed2gff.py -i $ds_rna_bed_path -o $ds_rna_gff_path -t $id_convert_table_path
python3 $preprocess_main_root/bed2gff.py -i $ds_canonical_bed_path -o $ds_canonical_gff_path -t $alt_region_id_table_path

if [ -e "$ds_region_fasta_path.fai" ]; then
    rm $ds_region_fasta_path.fai
fi

samtools faidx $ds_region_fasta_path

echo "Step 13: Write statistic data of GFF"

if [ ! -e "$saved_root/rna_stats/gff_analysis.log" ]; then
    bash  $bash_root/gff_analysis.sh -i $ds_rna_gff_path -f $ds_region_fasta_path -o $saved_root/rna_stats -r $ds_region_table_path -s new_id
fi

if [ ! -e "$saved_root/canonical_stats/gff_analysis.log" ]; then
    bash  $bash_root/gff_analysis.sh -i $ds_canonical_gff_path -f $ds_region_fasta_path -o $saved_root/canonical_stats -r $ds_region_table_path -s new_id
fi

echo "End process_main.sh"
exit 0
