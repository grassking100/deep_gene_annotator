#!/bin/bash
## function print usage
usage(){
 echo "Usage: The pipeline process data"
 echo "  Arguments:"
 echo "    -p  <string>  Path of preserved bed"
 echo "    -b  <string>  Path of background bed"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -t  <string>  Gene and transcript id conversion table path"
 echo "    -s  <string>  Source name"
 echo "    -o  <string>  Directory for output result"
 echo "  Options:"
 echo "    -m  <bool>    Merge regions which are overlapped                          [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash process_data.sh -u 1000 -d 1000 -g genome.fasta -p example.bed -b background.bed -t convert.tsv -o result -s Arabidopsis_1"
 echo ""
}

while getopts p:b:u:d:g:t:s:o:mh option
 do
  case "${option}"
  in
   p )preserved_bed_path=$OPTARG;;
   b )background_bed_path=$OPTARG;;
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   g )genome_path=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   s )source_name=$OPTARG;;
   o )saved_root=$OPTARG;;
   m )merge_overlapped=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$preserved_bed_path" ]; then
    echo "Missing option -p"
    usage
    exit 1
fi

if [ ! "$background_bed_path" ]; then
    echo "Missing option -b"
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

if [ ! "$id_convert_table_path" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

if [ ! "$source_name" ]; then
    echo "Missing option -s"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$merge_overlapped" ]; then
    merge_overlapped=false
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
visual_main_root=$script_root/sequence_annotation/visual
genome_handler_main_root=$script_root/sequence_annotation/genome_handler
rm -rf $saved_root
mkdir -p $saved_root


kwargs_path=$saved_root/process_kwargs.csv
echo "name,value" > $kwargs_path
echo "preserved_bed_path,$preserved_bed_path" >> $kwargs_path
echo "background_bed_path,$background_bed_path" >> $kwargs_path
echo "upstream_dist,$upstream_dist" >> $kwargs_path
echo "downstream_dist,$downstream_dist" >> $kwargs_path
echo "genome_path,$genome_path" >> $kwargs_path
echo "id_convert_table_path,$id_convert_table_path" >> $kwargs_path
echo "source_name,$source_name" >> $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path
echo "merge_overlapped,$merge_overlapped" >> $kwargs_path

echo "Step 1: Remove wanted RNAs which are overlapped with unwanted data in specific distance on both strands"
cleaning_root=$saved_root/cleaning
mkdir -p $cleaning_root
recurrent_cleaned_bed_path=$cleaning_root/recurrent_cleaned.bed
recurrent_cleaned_gene_bed_path=$cleaning_root/recurrent_cleaned_gene.bed
python3 $preprocess_main_root/recurrent_cleaner.py -b $background_bed_path -p $preserved_bed_path -f $genome_path.fai -u $upstream_dist -d $downstream_dist -s $cleaning_root -t $id_convert_table_path > $cleaning_root/recurrent.log

awk -F '\t' -v OFS='\t' '{print($1,$2,$3,$4,$5,$6)}' $recurrent_cleaned_bed_path > $recurrent_cleaned_gene_bed_path
python3 $preprocess_main_root/convert_bed_id.py -i $recurrent_cleaned_gene_bed_path -t $id_convert_table_path -o $recurrent_cleaned_gene_bed_path

echo "Step 2: Create regions around wanted RNAs on both strand"
region_root=$saved_root/region
coordinate_unique_gene_bed_path=$region_root/coordinate_unique_gene.bed
potential_region_bed_path=$region_root/potential_region.bed
both_strand_potential_region_bed_path=$region_root/both_strand_potential_region.bed
mkdir -p $region_root

python3 $preprocess_main_root/remove_coordinate_duplicated_bed.py -i $recurrent_cleaned_gene_bed_path -o $coordinate_unique_gene_bed_path
bedtools slop -s -i $coordinate_unique_gene_bed_path -g $genome_path.fai -l $upstream_dist -r $downstream_dist > $potential_region_bed_path
awk -F '\t' -v OFS='\t' '{print($1,$2,$3,".",".","+");print($1,$2,$3,".",".","-")}' $potential_region_bed_path \
> $both_strand_potential_region_bed_path

echo "Step 3: Get the number of region covering gene and filter the data by the number"
region_bed_path=$region_root/region.bed
bash $bash_root/gene_to_region_mapping.sh -i $both_strand_potential_region_bed_path -d $coordinate_unique_gene_bed_path \
-o $region_root/gene_region_mapping.bed -c $region_root/gene_region_mapping.count

if  $merge_overlapped ; then
    echo "Merge region on same strand"
    bash $bash_root/sort_merge.sh -i $region_root/gene_region_mapping.bed -o $region_bed_path -s
else
    echo "Filter region with region_gene_count_filter"
    python3 $preprocess_main_root/region_gene_count_filter.py -i $region_root/gene_region_mapping.bed \
    -r $coordinate_unique_gene_bed_path -o $region_bed_path -e $region_root/discarded_region.bed \
    -u $upstream_dist -d $downstream_dist
fi

echo "Step 4: Select region which its sequence only has A, T, C, and G"
region_fasta_path=$region_root/region.fasta
result_root=$saved_root/result
selected_region_bed_path=$result_root/selected_region.bed
mkdir -p $result_root
bedtools getfasta -name -fi $genome_path -bed $region_bed_path -fo $region_fasta_path
python3 $preprocess_main_root/get_cleaned_code_bed.py -b $region_bed_path \
-f $region_fasta_path -o $selected_region_bed_path -d $region_root/dirty_selected_region.bed

echo "Step 5: Get RNAs in selected regions"
final_gene_id_path=$result_root/gene.id
origin_rna_bed_path=$result_root/origin_rna.bed
python3 $preprocess_main_root/get_id.py -i $selected_region_bed_path -o $final_gene_id_path
python3 $preprocess_main_root/get_subbed.py -i $recurrent_cleaned_bed_path -d $final_gene_id_path \
-o $origin_rna_bed_path -t $id_convert_table_path

echo "Step 6: Rename region and get fasta"
echo "Caution: The dataset's strand which is single-strand data is indicated its location of origin's strand not new strand!!!"
region_table_path=$result_root/region_rename_table.tsv
selected_region_path=$result_root/selected_region.fasta
python3 $preprocess_main_root/rename_bed_id.py -i $selected_region_bed_path -p region -t $region_table_path \
-o $selected_region_bed_path --use_strand --record_coord_id
bedtools getfasta -s -name -fi $genome_path -bed $selected_region_bed_path -fo $selected_region_path
if [ -e "$selected_region_path.fai" ]; then
    rm $selected_region_path.fai
fi
samtools faidx $selected_region_path

echo "Step 7: Redefine coordinate based on region data"
rna_bed_path=$result_root/rna.bed
python3 $preprocess_main_root/redefine_coordinate.py -i $origin_rna_bed_path -t $region_table_path -o $rna_bed_path

echo "Step 8: Create data about gene structure and alternative status"
canonical_bed_path=$result_root/canonical.bed
canonical_gff_path=$result_root/canonical.gff3
canonical_h5_path=$result_root/canonical.h5
alt_region_gff_path=$result_root/alt_region.gff3
alt_region_h5_path=$result_root/alt_region.h5
alt_region_id_table_path=$result_root/alt_region_id_table.tsv
python3 $preprocess_main_root/convert_transcript_to_gene_with_alt_status_gff.py -i $rna_bed_path \
-o $alt_region_gff_path -t $id_convert_table_path
python3 $preprocess_main_root/create_gene_bed_from_exon_gff.py -i $alt_region_gff_path -o $canonical_bed_path
python3 $preprocess_main_root/get_id_table.py -i $alt_region_gff_path -o $alt_region_id_table_path
python3 $preprocess_main_root/bed2gff.py -i $canonical_bed_path -o $canonical_gff_path -t $alt_region_id_table_path
if $remove_alt_site ; then
    python3 $preprocess_main_root/create_ann_genome.py -i $canonical_gff_path -r $selected_region_bed_path -o $canonical_h5_path \
    -s source_name --discard_alt_region --discard_UTR_CDS
else
    python3 $preprocess_main_root/create_ann_genome.py -i $canonical_gff_path -r $selected_region_bed_path -o $canonical_h5_path \
    -s source_name  --discard_UTR_CDS
fi
python3 $preprocess_main_root/create_ann_genome.py -i $alt_region_gff_path \
-r $selected_region_bed_path -o $alt_region_h5_path -s source_name --with_alt_region --with_alt_site_region

python3 $genome_handler_main_root/ann_seqs_summary.py -i $canonical_h5_path -o $result_root/region_length

echo "Step 9: Create double-strand data"
double_strand_root=$result_root/double_strand
mkdir -p $double_strand_root
ds_region_table_path=$double_strand_root/region_rename_table_double_strand.tsv
ds_region_bed_path=$double_strand_root/selected_region_double_strand.bed
ds_region_fasta_path=$double_strand_root/selected_region_double_strand.fasta
ds_rna_bed_path=$double_strand_root/rna_double_strand.bed
ds_rna_gff_path=$double_strand_root/rna_double_strand.gff3
ds_canonical_bed_path=$double_strand_root/canonical_double_strand.bed
ds_canonical_gff_path=$double_strand_root/canonical_double_strand.gff3
python3 $preprocess_main_root/rename_bed_id.py -i $selected_region_bed_path -p region -t $ds_region_table_path \
-o $ds_region_bed_path --ignore_output_strand
bedtools getfasta -name -fi $genome_path -bed $ds_region_bed_path -fo $ds_region_fasta_path
python3 $preprocess_main_root/rename_chrom.py -i $rna_bed_path -t $ds_region_table_path -o $ds_rna_bed_path
python3 $preprocess_main_root/rename_chrom.py -i $canonical_bed_path -t $ds_region_table_path -o $ds_canonical_bed_path
python3 $preprocess_main_root/bed2gff.py -i $ds_rna_bed_path -o $ds_rna_gff_path -t $id_convert_table_path
python3 $preprocess_main_root/bed2gff.py -i $ds_canonical_bed_path -o $ds_canonical_gff_path -t $alt_region_id_table_path
if [ -e "$ds_region_fasta_path.fai" ]; then
    rm $ds_region_fasta_path.fai
fi
samtools faidx $ds_region_fasta_path

echo "Step 10: Write statistic data"
num_input_RNAs=$(wc -l < $preserved_bed_path )
num_recurrent=$(wc -l < $recurrent_cleaned_bed_path )
num_region_region=$(wc -l < $region_bed_path )
num_cleaned_region=$(wc -l < $selected_region_bed_path )
num_dirty_region=$(wc -l < $region_root/dirty_selected_region.bed )
num_final_rna=$(wc -l < $rna_bed_path )
num_final_region=$(wc -l < $selected_region_bed_path )
num_canonical=$(wc -l < $canonical_bed_path )

echo "The number of input RNAs: $num_input_RNAs" >> $saved_root/count.stats
echo "The number of recurrent-cleaned RNAs: $num_recurrent" >> $saved_root/count.stats
echo "The number of regions: $num_region_region" >> $saved_root/count.stats
echo "The number of cleaned regions: $num_cleaned_region" >> $saved_root/count.stats
echo "The number of dirty regions: $num_dirty_region" >> $saved_root/count.stats
echo "The number of final regions: $num_final_region" >> $saved_root/count.stats
echo "The number of cleaned genes in valid regions: $num_canonical" >> $saved_root/count.stats
echo "The number of cleaned RNAs in valid regions: $num_final_rna" >> $saved_root/count.stats


echo "Step 11: Write statistic data of GFF"

if [ ! -e "$saved_root/rna_stats/gff_analysis.log" ]; then
    bash  $bash_root/gff_analysis.sh -i $ds_rna_gff_path -f $ds_region_fasta_path -o $saved_root/rna_stats \
    -r $ds_region_table_path -s new_id
fi

if [ ! -e "$saved_root/canonical_stats/gff_analysis.log" ]; then
    bash  $bash_root/gff_analysis.sh -i $ds_canonical_gff_path -f $ds_region_fasta_path -o $saved_root/canonical_stats \
    -r $ds_region_table_path -s new_id
fi
