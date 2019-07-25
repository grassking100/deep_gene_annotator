#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating annotation data"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -w  <int>     Downstream distance"
 echo "    -r  <string>  Directory for Arabidopsis_thaliana"
 echo "    -o  <string>  Directory for output"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -t  <int>     Radius of Transcription start sites                [default: 100]"
 echo "    -d  <int>     Radius of Donor sites    [default: 100]"
 echo "    -a  <int>     Radius of Accept sites   [default: 100]"
 echo "    -c  <int>     Radius of Cleavage sites                           [default: 100]"
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_main.sh -u 10000 -w 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:w:r:o:t:c:d:a:s:h option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   w )downstream_dist=$OPTARG;;
   r )root=$OPTARG;;
   o )saved_root=$OPTARG;;
   t )tss_radius=$OPTARG;;
   c )cleavage_radius=$OPTARG;;
   d )donor_radius=$OPTARG;;
   a )accept_radius=$OPTARG;;
   s )source_name=$OPTARG;;
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
if [ ! "$root" ]; then
    echo "Missing option -r"
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

if [ ! "$tss_radius" ]; then
    tss_radius="100"
fi
if [ ! "$cleavage_radius" ]; then
    cleavage_radius="100"
fi
if [ ! "$donor_radius" ]; then
    donor_radius="100"
fi
if [ ! "$accept_radius" ]; then
    accept_radius="100"
fi

result_root=$saved_root/result
processed_root=$saved_root/processed
cleaning_root=$saved_root/cleaning
fasta_root=$saved_root/fasta
stats_root=$saved_root/stats
samtools=/home/samtools-1.9/samtools
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
sa_root=$bash_root/../..
src_root=$sa_root/sequence_annotation/gene_info

echo "Start of program"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root
mkdir -p $processed_root
mkdir -p $result_root
mkdir -p $cleaning_root
mkdir -p $fasta_root
mkdir -p $stats_root

#Set parameter
fai=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta.fai
genome_file=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
bed_target_path=$root/raw_data/Araport11_GFF3_genes_transposons.201606_repair_2019_07_21.bed
biomart_path=$root/raw_data/biomart_araport_11_gene_info_2018_11_27.csv
gro_1=$root/raw_data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
drs=$root/raw_data/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv
peptide_path=$root/raw_data/Araport11_genes.201606.pep.fasta
id_convert=$processed_root/id_convert.tsv

echo "Step 2: Preprocess raw data"
#Preprocess
python3 $src_root/preprocess_raw_data.py --output_root $processed_root --bed_path $bed_target_path \
--biomart_path $biomart_path --gro_1 $gro_1 --gro_2 $gro_2 --cs_path $drs

python3 $src_root/get_most_UTR.py -b $processed_root/valid_official_coding.bed -s $processed_root

echo "Step 3: Classify GRO and DRS data to belongs sites"
python3 $src_root/classify_sites.py -o $processed_root/valid_official_coding.bed  -g $processed_root/valid_gro.tsv -c $processed_root/valid_cleavage_site.tsv -s $processed_root -u $upstream_dist -d $downstream_dist \
-f $processed_root/most_five_UTR.tsv -t $processed_root/most_three_UTR.tsv > $processed_root/classify.log

echo "Step 4: Clean sites data"
python3 $src_root/consist_sites.py --ig $processed_root/inner_gro_sites.tsv \
--ic $processed_root/inner_cleavage_sites.tsv --lg $processed_root/long_dist_gro_sites.tsv \
--lc $processed_root/long_dist_cleavage_sites.tsv --tg $processed_root/transcript_gro_sites.tsv \
--tc $processed_root/transcript_cleavage_sites.tsv -s $processed_root

echo "Step 5: Create coordiante data based on origin data and site data"
python3 $src_root/create_coordinate_data.py -g $processed_root/safe_merged_gro_sites.tsv -c $processed_root/safe_merged_cleavage_sites.tsv -i $id_convert  -o $processed_root/consist_data.tsv
python3 $src_root/create_coordinate_bed.py -c $processed_root/consist_data.tsv -b $processed_root/valid_official_coding.bed -i $id_convert -o $processed_root/coordinate_consist.bed

echo "Step 6: Remove overlapped data"
python3 $src_root/nonoverlap_filter.py -c $processed_root/coordinate_consist.bed -i $id_convert -s $cleaning_root

echo "Step 7: Remove overlap data with specific distance"
python3 $src_root/recurrent_cleaner.py -r $processed_root/official.bed -c $cleaning_root/nonoverlap.bed -f $fai -u $upstream_dist -d $downstream_dist -s $cleaning_root -i $id_convert > $cleaning_root/recurrent.log

echo "Step 8: Get fasta of region around TSSs, CAs, donor sites, accept sites and get fasta of peptide and cDNA"
echo "       , and write statistic data"
echo "bash $bash_root/bed_analysis.sh -i $cleaning_root/recurrent_cleaned.bed -g $genome_file -p $peptide_path -o $fasta_root -s $stats_root"
bash $bash_root/bed_analysis.sh -i $cleaning_root/recurrent_cleaned.bed -g $genome_file -p $peptide_path -o $fasta_root -s $stats_root

echo "Step 9: Create data around selected region"
cp $cleaning_root/recurrent_cleaned.bed $result_root/cleaned.bed
bash $bash_root/get_region.sh -i $result_root/cleaned.bed -f $fai -u $upstream_dist -d $downstream_dist -o $result_root/selected_region.bed
python3 $src_root/rename_bed.py -i $result_root/selected_region.bed -p seq -t $result_root/rename_table.tsv -o $result_root/selected_region.bed

bedtools getfasta -name -fi $genome_file -bed $result_root/selected_region.bed -fo $result_root/selected_region.fasta 
$samtools faidx $result_root/selected_region.fasta

python3 $src_root/redefine_coordinate.py -i $result_root/cleaned.bed -t $result_root/rename_table.tsv -r $result_root/selected_region.bed -o $result_root/cleaned.bed

#python3 $src_root/create_ann_region.py -i $result_root/cleaned.bed -f $result_root/selected_region.fasta.fai -s $source_name -o $result_root/selected_region.h5

python3 $src_root/bed2gff.py $result_root/cleaned.bed $result_root/cleaned.gff $id_convert

#mv $result_root/result.gff $result_root/output.gff
#python3 $src_root/gff2bed.py -i $result_root/output.gff -o $result_root/output.bed -m true
#bedtools getfasta -s -fi $result_root/selected_region.fasta  -bed $result_root/output.bed -fo $result_root/output.fasta
echo "Step 10: Write statistic data"
num_valid_official=$(wc -l < $processed_root/valid_official_coding.bed )
num_gro=$(sed "1d" $processed_root/safe_merged_gro_sites.tsv | wc -l)
num_drs=$(sed "1d" $processed_root/safe_merged_cleavage_sites.tsv | wc -l)
num_consist=$(wc -l < $processed_root/coordinate_consist.bed )
num_nonoverlap=$(wc -l < $cleaning_root/nonoverlap.bed )
num_cleaned=$(wc -l < $result_root/cleaned.bed )
num_region=$(wc -l < $result_root/selected_region.bed )

echo "Valid coding mRNA count: $num_valid_official" > $stats_root/count.stats
echo "Matched GRO sites count: $num_gro" >> $stats_root/count.stats
echo "Matched DRS sites count: $num_drs" >> $stats_root/count.stats
echo "The number of mRNAs with both GRO and DRS sites supported: $num_consist" >> $stats_root/count.stats
echo "The number of mRNAs which are not overlap with each other: $num_nonoverlap" >> $stats_root/count.stats
echo "The number of mRNAs after recurrent cleaning: $num_cleaned" >> $stats_root/count.stats
echo "The number of regions: $num_region" >> $stats_root/count.stats

echo "End of program"
exit 0