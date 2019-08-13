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
 echo "    -d  <int>     Radius of Donor sites                              [default: 100]"
 echo "    -a  <int>     Radius of Accept sites                             [default: 100]"
 echo "    -c  <int>     Radius of Cleavage sites                           [default: 100]"
 echo "    -f  <bool>    Filter with ORF                                    [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_main.sh -u 10000 -w 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:w:r:o:t:c:d:a:s:f:h option
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
   f )filter_orf=$OPTARG;;
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
if [ ! "$filter_orf" ]; then
    filter_orf=false
fi

processed_root=$saved_root/processed
samtools=/home/samtools-1.9/samtools
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
gene_info_root=$script_root/sequence_annotation/gene_info

echo "Start of program"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root
mkdir -p $processed_root

#Set parameter
genome_path=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
bed_target_path=$root/raw_data/Araport11_GFF3_genes_transposons.201606_repair_2019_07_21.bed
biomart_path=$root/raw_data/biomart_araport_11_gene_info_2018_11_27.csv
gro_1=$root/raw_data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
drs=$root/raw_data/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv
peptide_path=$root/raw_data/Araport11_genes.201606.pep.fasta
fai_path=$genome_path.fai
id_convert=$processed_root/id_convert.tsv
background_bed_path=$processed_root/official.bed
echo "Step 2: Preprocess raw data"
#Preprocess
python3 $gene_info_root/preprocess_raw_data.py --output_root $processed_root --bed_path $bed_target_path \
--biomart_path $biomart_path --gro_1 $gro_1 --gro_2 $gro_2 --cs_path $drs

python3 $gene_info_root/get_most_UTR.py -b $processed_root/valid_official_coding.bed -s $processed_root

echo "Step 3: Classify GRO and DRS data to belonging sites"
python3 $gene_info_root/classify_sites.py -o $processed_root/valid_official_coding.bed  -g $processed_root/valid_gro.tsv -c $processed_root/valid_cleavage_site.tsv -s $processed_root -u $upstream_dist -d $downstream_dist \
-f $processed_root/most_five_UTR.tsv -t $processed_root/most_three_UTR.tsv > $processed_root/classify.log

echo "Step 4: Clean sites data"
python3 $gene_info_root/consist_sites.py --ig $processed_root/inner_gro_sites.tsv \
--ic $processed_root/inner_cleavage_sites.tsv --lg $processed_root/long_dist_gro_sites.tsv \
--lc $processed_root/long_dist_cleavage_sites.tsv --tg $processed_root/transcript_gro_sites.tsv \
--tc $processed_root/transcript_cleavage_sites.tsv -s $processed_root

echo "Step 5: Create coordiante data based on origin data and site data"
python3 $gene_info_root/create_coordinate_data.py -g $processed_root/safe_merged_gro_sites.tsv -c $processed_root/safe_merged_cleavage_sites.tsv -i $id_convert  -o $processed_root/consist_data.tsv
python3 $gene_info_root/create_coordinate_bed.py -c $processed_root/consist_data.tsv -b $processed_root/valid_official_coding.bed -i $id_convert -o $processed_root/coordinate_consist.bed -s $filter_orf

num_valid_official=$(wc -l < $processed_root/valid_official_coding.bed )
num_gro=$(sed "1d" $processed_root/safe_merged_gro_sites.tsv | wc -l)
num_drs=$(sed "1d" $processed_root/safe_merged_cleavage_sites.tsv | wc -l)
num_consist=$(wc -l < $processed_root/coordinate_consist.bed )
   
echo "Valid coding mRNA count: $num_valid_official" > $processed_root/preprocess.stats
echo "Matched GRO sites count: $num_gro" >> $processed_root/preprocess.stats
echo "Matched DRS sites count: $num_drs" >> $processed_root/preprocess.stats
echo "The number of mRNAs with both GRO and DRS sites supported and are passed by filter: $num_consist" >> $processed_root/preprocess.stats

bash $bash_root/process_data.sh -u $upstream_dist -w $downstream_dist -g $genome_path -i $processed_root/coordinate_consist.bed -o $saved_root -s $source_name -p $id_convert -b $background_bed_path -f $filter_orf -t $tss_radius -c $cleavage_radius -d $donor_radius -a $accept_radius

split_root=$saved_root/result/split
mkdir -p $split_root

python3 sequence_annotation/sequence_annotation/gene_info/split.py --region_bed_path $saved_root/result/selected_region.bed --region_rename_table_path $saved_root/result/region_rename_table.tsv --fai_path $fai_path --splitted_id_root $split_root

for path in $(find $saved_root/result/split/* -name '*.tsv');
do
    file_name=$(basename $path)
    file_name="${file_name%.*}"
    python3 $gene_info_root/get_subbed.py -i $saved_root/result/canonical.bed -d $split_root/$file_name.tsv \
    -o $split_root/$file_name.bed --query_column chr
    python3 $gene_info_root/get_GlimmerHMM_cds_file.py -i $split_root/$file_name.bed -o $split_root/$file_name.cds
    python3 $gene_info_root/bed2gff.py -i $split_root/$file_name.bed -o $split_root/$file_name.gff
    python3 $gene_info_root/get_subfasta.py -i $saved_root/result/selected_region.fasta -d $split_root/$file_name.tsv -o $split_root/$file_name.fasta 
done

exit 0