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
 echo "    -m  <bool>    Merge regions which are overlapped                        [default: false]"
 echo "    -c  <bool>    Remove gene with altenative donor site and acceptor site    [default: false]"
 echo "    -t  <string>  Gene and mRNA id converted table path, it will be created if it doesn't be provided"
 echo "    -b  <string>  Path of background bed, it will be set by input path if it doesn't be provided"
 echo "    -h            Print help message and exit"
 echo "Example: bash process_data.sh -u 10000 -d 10000 -g /home/io/genome.fasta -i /home/io/example.bed -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:g:i:o:s:t:b:mcxh option
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

if [ ! "$create_id" ]; then
    create_id=false
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


#root
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
result_root=$saved_root/result
cleaning_root=$saved_root/cleaning
fasta_root=$saved_root/fasta
stats_root=$saved_root/stats
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
result_canonical_bed_path=$result_root/canonical.bed
region_table_path=$result_root/region_rename_table.tsv

echo "Start process_main.sh"
#Create folder
echo "Step 1: Create folder and create id table"

rm -rf $saved_root
rm -rf $result_root
rm -rf $region_selected_root
rm -rf $cleaning_root
rm -rf $fasta_root
rm -rf $stats_root

mkdir -p $saved_root
mkdir -p $result_root
mkdir -p $cleaning_root
mkdir -p $region_selected_root
mkdir -p $fasta_root
mkdir -p $stats_root

cp $bed_path $saved_root/input.bed


if [ ! "$id_convert_table_path" ]; then
    id_convert_table_path=$saved_root/id_table.tsv
    python3 $preprocess_main_root/create_id_table_by_coord.py -i $saved_root/input.bed -o $id_convert_table_path -p gene
    
fi

echo "Step 2: Remove overlapped gene on the same strand"
python3 $preprocess_main_root/nonoverlap_filter.py -i $saved_root/input.bed -s $cleaning_root \
--use_strand --id_convert_path $id_convert_table_path 

echo "Step 2 (Optional): Remove transcript which its gene has alternative donor or acceptor site"
if $remove_alt_site ; then
    python3 $preprocess_main_root/remove_alt_site.py -i $cleaning_root/nonoverlap.bed \
    -o $cleaning_root/pass_filter_temp.bed -t $id_convert_table_path
else
    cp $cleaning_root/nonoverlap.bed $cleaning_root/pass_filter_temp.bed
fi

echo "Step 2 (Optional): Remove transcript which its gene has non-coding transcript"
if $remove_non_coding ; then
    python3 $preprocess_main_root/belonging_gene_coding_filter.py -i $cleaning_root/pass_filter_temp.bed -t $id_convert_table_path -o $cleaning_root/pass_filter.bed
else
    cp $cleaning_root/pass_filter_temp.bed $cleaning_root/pass_filter.bed
fi

rm $cleaning_root/pass_filter_temp.bed

echo "Step 3: Remove wanted RNAs which are overlapped with unwanted data in specific distance on both strands"
python3 $preprocess_main_root/recurrent_cleaner.py -r $background_bed_path -c $cleaning_root/pass_filter.bed -f $genome_fai \
-u $upstream_dist -d $downstream_dist -s $cleaning_root --id_convert_path $id_convert_table_path > $cleaning_root/recurrent.log

echo "Step 4: Create regions around wanted RNAs on both strand"
cp $cleaning_root/recurrent_cleaned.bed $selected_rna_bed_path

if [ -e "$result_root/selected_region.fasta.fai" ]; then
    rm $result_root/selected_region.fasta.fai
fi

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

echo "Step 6: Select region which its sequence only has A, T, C, and G"
bedtools getfasta -name -fi $genome_path -bed $region_selected_root/selected_region.bed -fo $region_selected_root/temp.fasta
python3 $preprocess_main_root/get_cleaned_code_bed.py -b $region_selected_root/selected_region.bed -f $region_selected_root/temp.fasta -o $region_selected_root/cleaned_selected_region.bed -d $region_selected_root/dirty_selected_region.bed
rm $region_selected_root/temp.fasta
cp $region_selected_root/cleaned_selected_region.bed $region_bed_path

echo "Step 7: Get RNAs in selected regions"
python3 $preprocess_main_root/get_id.py -i $region_bed_path -o $result_root/gene.id
python3 $preprocess_main_root/get_subbed.py -i $selected_rna_bed_path -d $result_root/gene.id \
-o $rna_bed_path -t $id_convert_table_path
rm $result_root/gene.id

echo "Step 8: Rename region and get fasta"

python3 $preprocess_main_root/rename_id_by_coordinate.py -i $region_bed_path -p region -t $region_table_path \
-o $region_bed_path --use_strand --coord_id_as_old_id

bedtools getfasta -s -name -fi $genome_path -bed $region_bed_path -fo $result_root/selected_region.fasta

samtools faidx $result_root/selected_region.fasta

cp $rna_bed_path $result_root/origin_rna.bed

python3 $preprocess_main_root/redefine_coordinate.py -i $rna_bed_path -t $region_table_path -o $rna_bed_path

echo "Step 9: Canonical path decoding"
python3 $preprocess_main_root/convert_transcript_bed_to_gene_gff.py -i $rna_bed_path -o $result_root/alt_region.gff -t $id_convert_table_path

if $remove_alt_site ; then
    python3 $preprocess_main_root/create_canonical_bed.py -i $result_root/alt_region.gff -o $result_canonical_bed_path -t $preprocess_main_root/standard_codon.tsv -f $result_root/selected_region.fasta
else
    python3 $preprocess_main_root/create_canonical_bed.py -i $result_root/alt_region.gff -o $result_canonical_bed_path -t $preprocess_main_root/standard_codon.tsv -f $result_root/selected_region.fasta --alt_site_region_as_exon
fi

python3 $preprocess_main_root/bed2gff.py -i $result_canonical_bed_path -o $result_root/canonical.gff

if $remove_alt_site ; then
    python3 $preprocess_main_root/create_ann_genome.py -i $result_root/canonical.gff -r $region_bed_path -o $result_root/canonical.h5 -s source_name --discard_alt_region --discard_UTR_CDS
    
else
    python3 $preprocess_main_root/create_ann_genome.py -i $result_root/canonical.gff -r $region_bed_path -o $result_root/canonical.h5 -s source_name  --discard_UTR_CDS
fi

python3 $preprocess_main_root/create_ann_genome.py -i $result_root/alt_region.gff \
    -r $region_bed_path -o $result_root/alt_region.h5 -s source_name --with_alt_region --with_alt_site_region

echo "Step 10: Write statistic data"
num_input_RNAs=$(wc -l < $bed_path )
num_background_RNAs=$(wc -l < $background_bed_path )
num_nonoverlap=$(wc -l < $cleaning_root/nonoverlap.bed )
num_pass_filter=$(wc -l < $cleaning_root/pass_filter.bed )
num_recurrent=$(wc -l < $cleaning_root/recurrent_cleaned.bed )
num_cleaned_region=$(wc -l < $region_selected_root/selected_region.bed )
num_dirty_region=$(wc -l < $region_selected_root/dirty_selected_region.bed )
num_final_rna=$(wc -l < $rna_bed_path )
num_final_region=$(wc -l < $region_bed_path )
num_canonical=$(wc -l < $result_canonical_bed_path )

echo "The number of background RNAs: $num_background_RNAs" > $stats_root/count.stats
echo "The number of selected to input RNAs: $num_input_RNAs" >> $stats_root/count.stats
echo "The number of RNAs which are not overlap with each other: $num_nonoverlap" >> $stats_root/count.stats
echo "The number of RNAs passed filter: $num_pass_filter" >> $stats_root/count.stats
echo "The number of recurrent-cleaned RNAs: $num_recurrent" >> $stats_root/count.stats
echo "The number of cleaned regions: $num_cleaned_region" >> $stats_root/count.stats
echo "The number of dirty regions: $num_dirty_region" >> $stats_root/count.stats
echo "The number of final regions: $num_final_region" >> $stats_root/count.stats
echo "The number of cleaned genes in valid regions: $num_canonical" >> $stats_root/count.stats
echo "The number of cleaned RNAs in valid regions: $num_final_rna" >> $stats_root/count.stats

echo "Step 11: Create double-strand data"
ds_region_table_path=$result_root/region_rename_table_both_strand.tsv
ds_region_bed_path=$result_root/selected_region_both_strand.bed
ds_region_fasta_path=$result_root/selected_region_both_strand.fasta
ds_rna_bed_path=$result_root/rna_both_strand.bed
ds_canonical_bed_path=$result_root/canonical_both_strand.bed

python3 $preprocess_main_root/rename_id_by_coordinate.py -i $region_bed_path -p region -t $ds_region_table_path -o $ds_region_bed_path
bedtools getfasta -name -fi $genome_path -bed $ds_region_bed_path -fo $ds_region_fasta_path
python3 $preprocess_main_root/rename_bed_chrom.py -i $rna_bed_path -t $ds_region_table_path -o $ds_rna_bed_path
python3 $preprocess_main_root/rename_bed_chrom.py -i $result_canonical_bed_path -t $ds_region_table_path -o $ds_canonical_bed_path

echo "Step 12: Get fasta of region around TSSs, CAs, donor sites, acceptor sites and get fasta of peptide and cDNA"
echo "       , and write statistic data"
bash $bash_root/bed_analysis.sh -i $ds_rna_bed_path -f $ds_region_fasta_path \
-o $fasta_root -s $stats_root -c $downstream_dist 1> $stats_root/log.txt 2>&1

echo "End process_main.sh"
exit 0
