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

while getopts u:d:g:i:o:s:t:b:mh option
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
    echo "Use $bed_path as background_bed_path"
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
region_selected_root=$saved_root/region_selected

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
gene_info_root=$script_root/sequence_annotation/preprocess
genome_handler_root=$script_root/sequence_annotation/genome_handler
region_bed_path=$result_root/selected_region.bed
rna_bed_path=$result_root/rna.bed
genome_fai=$genome_path.fai
echo "Start process_main.sh"
#Create folder
echo "Step 1: Create folder"

rm -rf $saved_root
rm -rf $result_root
rm -rf region_selected_root
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

echo "Step 2: Remove overlapped gene on the same strand"
if [ ! "$id_convert_table_path" ]; then
    id_convert_table_path=$saved_root/id_table.tsv
    python3 $gene_info_root/create_id_table_by_coord.py -i $saved_root/input.bed -o $id_convert_table_path -p gene
    
fi

python3 $gene_info_root/nonoverlap_filter.py -i $saved_root/input.bed -s $cleaning_root --use_strand --id_convert_path $id_convert_table_path 

echo "Step 3: Remove wanted RNAs which are overlapped with unwanted data in specific distance on both strands"
python3 $gene_info_root/recurrent_cleaner.py -r $background_bed_path -c $cleaning_root/nonoverlap.bed -f $genome_fai \
-u $upstream_dist -d $downstream_dist -s $cleaning_root --id_convert_path $id_convert_table_path > $cleaning_root/recurrent.log

echo "Step 4: Create regions around wanted RNAs on both strand"
cp $cleaning_root/recurrent_cleaned.bed $rna_bed_path

if [ -e "$result_root/selected_region.fasta.fai" ]; then
    rm $result_root/selected_region.fasta.fai
fi

echo "Merge: $merge_overlapped"

#Get potential regions
gene_bed_path=$region_selected_root/gene.bed

awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,$4,$5,$6)
}' $rna_bed_path > $gene_bed_path

python3 $gene_info_root/convert_bed_id.py -i $gene_bed_path -t $id_convert_table_path -o $gene_bed_path
python3 $gene_info_root/remove_coordinate_duplicated_bed.py -i $gene_bed_path -o $gene_bed_path
bedtools slop -s -i $gene_bed_path -g $genome_fai -l $upstream_dist -r $downstream_dist > $region_selected_root/potential_region.bed

awk -F '\t' -v OFS='\t' '{
    print($1,$2,$3,".",".","+")
    print($1,$2,$3,".",".","-")
}' $region_selected_root/potential_region.bed > $region_selected_root/potential_region.temp && mv $region_selected_root/potential_region.temp $region_selected_root/potential_region.bed

bash $bash_root/gene_to_region_mapping.sh -i $region_selected_root/potential_region.bed -d $gene_bed_path -o $region_selected_root/rna_region_mapping.bed -s $region_selected_root/rna_region_mapping.count

if  $merge_overlapped ; then
    echo "Merge region on same strand"
    bash $bash_root/sort_merge.sh -i $region_selected_root/rna_region_mapping.bed -o $region_selected_root/selected_region.bed -s
else
    echo "Filter region with region_filter"
    python3 $gene_info_root/region_filter.py -i $region_selected_root/rna_region_mapping.bed -r $gene_bed_path -o $region_selected_root/selected_region.bed -e $region_selected_root/discarded_region.bed -u $upstream_dist -d $downstream_dist
fi

echo "Check regions have both strands"
mv $region_selected_root/selected_region.bed $region_selected_root/selected_region_before_strand_filter.bed
python3 $gene_info_root/both_strand_include_filter.py -i $region_selected_root/selected_region_before_strand_filter.bed -o $region_selected_root/selected_region.bed

#Get RNAs in selected regions
bash $bash_root/get_ids.sh -i $region_selected_root/selected_region.bed > $region_selected_root/selected_gene.id
python3 $gene_info_root/get_subbed.py -i $rna_bed_path -d $region_selected_root/selected_gene.id \
-o $rna_bed_path -t $id_convert_table_path
cp $region_selected_root/selected_region.bed $region_bed_path

echo "Step 5: Rename region and get fasta"

python3 $gene_info_root/rename_bed.py -i $region_bed_path -p region -t $result_root/region_rename_table.tsv -o $region_bed_path --use_strand

bedtools getfasta -name -fi $genome_path -bed $region_bed_path -fo $result_root/selected_region_no_strand.fasta
bedtools getfasta -s -name -fi $genome_path -bed $region_bed_path -fo $result_root/selected_region.fasta

samtools faidx $result_root/selected_region_no_strand.fasta
samtools faidx $result_root/selected_region.fasta

cp $rna_bed_path $result_root/origin_rna.bed

python3 $gene_info_root/redefine_coordinate.py -i $rna_bed_path -t $result_root/region_rename_table.tsv -o $rna_bed_path --use_strand

echo "Step 6: Canonical path decoding"
python3 $gene_info_root/path_decode.py -i $rna_bed_path -o $result_root/alt_region.gff -t $id_convert_table_path

python3 $gene_info_root/create_canonical_bed.py -i $result_root/alt_region.gff -o $result_root/canonical.bed \
-t $gene_info_root/standard_codon.tsv -f $result_root/selected_region.fasta

python3 $gene_info_root/bed2gff.py -i $result_root/canonical.bed -o $result_root/canonical.gff

python3 $gene_info_root/alt_anns_creator.py -i $result_root/alt_region.gff \
    -r $region_bed_path -o $result_root/alt_region.h5 -s source_name

echo "Step 7: Get fasta of region around TSSs, CAs, donor sites, accept sites and get fasta of peptide and cDNA"
echo "       , and write statistic data"

bash $bash_root/bed_analysis.sh -i $rna_bed_path -f $result_root/selected_region_no_strand.fasta -o $fasta_root -s $stats_root 1 > $stats_root/log.txt 2>&1

num_input_RNAs=$(wc -l < $bed_path )
num_background_RNAs=$(wc -l < $background_bed_path )
num_nonoverlap=$(wc -l < $cleaning_root/nonoverlap.bed )
num_recurrent=$(wc -l < $cleaning_root/recurrent_cleaned.bed )
num_cleaned=$(wc -l < $rna_bed_path )
num_region=$(wc -l < $region_bed_path )
num_canonical=$(wc -l < $result_root/canonical.bed )

echo "The number of background RNAs: $num_background_RNAs" > $stats_root/count.stats
echo "The number of selected to input RNAs: $num_input_RNAs" >> $stats_root/count.stats
echo "The number of RNAs which are not overlap with each other: $num_nonoverlap" >> $stats_root/count.stats
echo "The number of recurrent-cleaned RNAs: $num_recurrent" >> $stats_root/count.stats
echo "The number of cleaned gene in valid region: $num_canonical" >> $stats_root/count.stats
echo "The number of cleaned RNAs in valid region: $num_cleaned" >> $stats_root/count.stats
echo "The number of regions: $num_region" >> $stats_root/count.stats

echo "End process_main.sh"
exit 0
