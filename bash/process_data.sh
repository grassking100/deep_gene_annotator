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
 echo "    -m  <bool>    Merge regions which are overlapped                 [default: false]"
 echo "    -t  <string>  Gene and mRNA id converted table path, it will be created if it doesn't be provided"
 echo "    -b  <string>  Path of background bed, it will be set to input path if it doesn't be provided"
 echo "    -h            Print help message and exit"
 echo "Example: bash process_data.sh -u 10000 -d 10000 -g /home/io/genome.fasta -i /home/io/example.bed -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:g:i:o:s:t:bm:h option
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
    background_bed_path=$bed_path
fi

if [ ! "$create_id" ]; then
    create_id=false
fi

if [ ! "$merge_overlapped" ]; then
    merge_overlapped=false
fi

result_root=$saved_root/result
cleaning_root=$saved_root/cleaning
fasta_root=$saved_root/fasta
stats_root=$saved_root/stats
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
gene_info_root=$script_root/sequence_annotation/gene_info
genome_handler_root=$script_root/sequence_annotation/genome_handler

genome_fai=$genome_path.fai
echo "Start process_main.sh"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root
mkdir -p $result_root
mkdir -p $cleaning_root
mkdir -p $fasta_root
mkdir -p $stats_root

cp $bed_path $saved_root/input.bed

echo "Step 2: Remove overlapped gene on the same strand"
if [ ! "$id_convert_table_path" ]; then
    id_convert_table_path=$saved_root/id_table.tsv
    python3 $gene_info_root/create_id_table_by_coord.py -i $saved_root/input.bed -o $id_convert_table_path -p gene
    
fi


python3 $gene_info_root/nonoverlap_filter.py -i $saved_root/input.bed -s $cleaning_root --use_strand --id_convert_path $id_convert_table_path 

echo "Step 3: Remove wanted RNAs which are overlapped with unwanted data in specific distance on both strands"
python3 $gene_info_root/recurrent_cleaner.py -r $background_bed_path -c $cleaning_root/nonoverlap.bed -f $genome_fai \
-u $upstream_dist -d $downstream_dist -s $cleaning_root --id_convert_path $id_convert_table_path > $cleaning_root/recurrent.log

echo "Step 4: Create regions around wanted RNAs (ignore strand information)"
cp $cleaning_root/recurrent_cleaned.bed $result_root/cleaned.bed

if [ -e "$result_root/selected_region.fasta.fai" ]; then
    rm $result_root/selected_region.fasta.fai
fi

if  $merge_overlapped ; then
    bash $bash_root/get_region.sh -i $result_root/cleaned.bed -f $genome_fai -u $upstream_dist \
    -d $downstream_dist -o $result_root/selected_region.bed -r
    bash $bash_root/get_ids.sh -i $result_root/selected_region.bed > $result_root/selected_RNA.id
    python3 $gene_info_root/get_subbed.py -i $result_root/cleaned.bed -d $result_root/selected_RNA.id -o $result_root/cleaned.bed
else
    region_bed_path=$result_root/_extened.bed
    converted_bed_path=$result_root/_converted.bed
    bedtools slop -s -i $result_root/cleaned.bed -g $genome_fai -l $upstream_dist -r $downstream_dist > $region_bed_path
    python3 $gene_info_root/convert_bed_id.py -i $result_root/cleaned.bed -t $id_convert_table_path -o $converted_bed_path
    bash $bash_root/single_gene_region_filter.sh -i $region_bed_path -d $converted_bed_path -o $result_root/selected_region.bed
    bash $bash_root/get_ids.sh -i $result_root/selected_region.bed > $result_root/selected_RNA.id
    mv $result_root/cleaned.bed $result_root/cleaned.bed.temp 
    python3 $gene_info_root/get_subbed.py -i $result_root/cleaned.bed.temp -d $result_root/selected_RNA.id -o $result_root/cleaned.bed
    rm $converted_bed_path
    rm $region_bed_path
    rm $result_root/cleaned.bed.temp
fi

echo "Step 5: Rename region and get fasta"
python3 $gene_info_root/rename_bed.py -i $result_root/selected_region.bed -p region \
-t $result_root/region_rename_table.tsv -o $result_root/selected_region.bed

bedtools getfasta -name -fi $genome_path -bed $result_root/selected_region.bed -fo $result_root/selected_region.fasta
samtools faidx $result_root/selected_region.fasta
awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,$4"plus",$5,"+")
    print($1,$2,$3,$4"minus",$5,"-")
}' $result_root/selected_region.bed > $result_root/selected_region_strand.bed

bedtools getfasta -s -name -fi $genome_path -bed $result_root/selected_region_strand.bed -fo $result_root/selected_region_strand.fasta

python3 $gene_info_root/redefine_coordinate.py -i $result_root/cleaned.bed -t $result_root/region_rename_table.tsv \
-r $result_root/selected_region.bed -o $result_root/cleaned.bed

echo "Step 6: Canonical path decoding"
if [ ! -e "$result_root/alt_region.gff" ]; then
    python3 $gene_info_root/path_decode.py -i $result_root/cleaned.bed -o $result_root/alt_region.gff -t $id_convert_table_path
fi

python3 $gene_info_root/create_canonical_bed.py -i $result_root/alt_region.gff -o $result_root/canonical.bed \
-t $gene_info_root/standard_codon.tsv -f $result_root/selected_region.fasta

python3 $gene_info_root/bed2gff.py -i $result_root/canonical.bed -o $result_root/canonical.gff

if [ ! -e "$result_root/alt_region.h5" ]; then
    python3 $gene_info_root/alt_anns_creator.py -i $result_root/alt_region.gff \
    -f $result_root/selected_region.fasta.fai -o $result_root/alt_region.h5 -s source_name
fi

echo "Step 7: Get fasta of region around TSSs, CAs, donor sites, accept sites and get fasta of peptide and cDNA"
echo "       , and write statistic data"

bash $bash_root/bed_analysis.sh -i $result_root/cleaned.bed -f $result_root/selected_region.fasta -o $fasta_root -s $stats_root

num_input_mRNAs=$(wc -l < $bed_path )
num_background_mRNAs=$(wc -l < $background_bed_path )
num_nonoverlap=$(wc -l < $cleaning_root/nonoverlap.bed )
num_recurrent=$(wc -l < $cleaning_root/recurrent_cleaned.bed )
num_cleaned=$(wc -l < $result_root/cleaned.bed )
num_region=$(wc -l < $result_root/selected_region.bed )
num_canonical=$(wc -l < $result_root/canonical.bed )

echo "The number of background mRNAs: $num_background_mRNAs" > $stats_root/count.stats
echo "The number of selected to input mRNAs: $num_input_mRNAs" >> $stats_root/count.stats
echo "The number of mRNAs which are not overlap with each other: $num_nonoverlap" >> $stats_root/count.stats
echo "The number of recurrent-cleaned mRNAs: $num_recurrent" >> $stats_root/count.stats
echo "The number of cleaned gene in valid region: $num_canonical" >> $stats_root/count.stats
echo "The number of cleaned mRNAs in valid region: $num_cleaned" >> $stats_root/count.stats
echo "The number of regions: $num_region" >> $stats_root/count.stats

echo "End process_main.sh"
exit 0
