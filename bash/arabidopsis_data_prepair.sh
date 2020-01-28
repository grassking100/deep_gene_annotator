#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline prepair annotation data"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -r  <string>  Directory of Arabidopsis thaliana data"
 echo "    -o  <string>  Directory of output folder"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -i  <bool>    Remove regions which inner signal                [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_data_prepair.sh -u 10000 -d 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:r:o:s:ih option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   r )root=$OPTARG;;
   o )saved_root=$OPTARG;;
   s )source_name=$OPTARG;;
   i )remove_inner_end=true;;
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

if [ ! "$remove_inner_end" ]; then
    remove_inner_end=false
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess

echo "Start of program"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root

#Set parameter
genome_path=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
gro_1=$root/raw_data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
DRS_path=$root/raw_data/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv
peptide_path=$root/raw_data/Araport11_genes.201606.pep.fasta
official_gff_path=$root/raw_data/Araport11_GFF3_genes_transposons.201606.gff
###
id_convert_table_path=$saved_root/id_convert.tsv
repaired_gff_path=$saved_root/repaired_official.gff
processed_gff_path=$saved_root/processed.gff
processed_bed_path=$saved_root/processed.bed
echo "Step 2: Preprocess raw data"

if [ ! -e "$id_convert_table_path" ]; then
    python3 $preprocess_main_root/get_id_table.py -i $official_gff_path -o $id_convert_table_path
fi

if [ ! -e "$repaired_gff_path" ]; then
    python3 $preprocess_main_root/repair_gff.py -i $official_gff_path -o $repaired_gff_path -s $saved_root
fi

if [ ! -e "$processed_gff_path" ]; then
    python3 $preprocess_main_root/preprocess_gff.py -i $repaired_gff_path -o $processed_gff_path
fi

if [ ! -e "$processed_bed_path" ]; then
    python3 $preprocess_main_root/gff2bed.py -i $processed_gff_path -o $processed_bed_path
fi
    
python3 $preprocess_main_root/preprocess_raw_data.py --output_root $saved_root --bed_path $processed_bed_path \
--gro_1 $gro_1 --gro_2 $gro_2 --cs_path $DRS_path 

python3 $preprocess_main_root/get_external_UTR.py -b $saved_root/valid_official.bed -s $saved_root

echo "Step 3: Classify GRO and DRS data to belonging sites"
python3 $preprocess_main_root/classify_sites.py -o $saved_root/valid_official.bed  -g $saved_root/valid_gro.tsv -c $saved_root/valid_cleavage_site.tsv -s $saved_root -u $upstream_dist -d $downstream_dist \
-f $saved_root/external_five_UTR.tsv -t $saved_root/external_three_UTR.tsv > $saved_root/classify.log

echo "Step 4: Get maximize signal sites data located on external UTR"
command="$preprocess_main_root/consist_sites.py --ig $saved_root/inner_gro_sites.tsv \
--ic $saved_root/inner_cleavage_sites.tsv --lg $saved_root/long_dist_gro_sites.tsv \
--lc $saved_root/long_dist_cleavage_sites.tsv -s $saved_root --tg $saved_root/transcript_gro_sites.tsv \
--tc $saved_root/transcript_cleavage_sites.tsv"

if $remove_inner_end; then
    command="${command} --remove_inner_end"
fi
    
python3 $command

echo "Step 5: Create coordiante data based on origin data and site data"
python3 $preprocess_main_root/create_coordinate_data.py -g $saved_root/safe_gro_sites.tsv -c $saved_root/safe_cleavage_sites.tsv -t $id_convert_table_path --single_start_end -o $saved_root/coordinate_data.tsv

python3 $preprocess_main_root/create_coordinate_bed.py -i $saved_root/valid_official.bed \
-c $saved_root/coordinate_data.tsv -t $id_convert_table_path -o $saved_root/coordinate_consist.bed

num_valid_official=$(wc -l < $saved_root/valid_official.bed )
num_gro=$(sed "1d" $saved_root/safe_gro_sites.tsv | wc -l)
num_drs=$(sed "1d" $saved_root/safe_cleavage_sites.tsv | wc -l)
num_consist=$(wc -l < $saved_root/coordinate_consist.bed )
   
echo "Selected mRNA count: $num_valid_official" > $saved_root/preprocess.stats
echo "Matched GRO sites count: $num_gro" >> $saved_root/preprocess.stats
echo "Matched DRS sites count: $num_drs" >> $saved_root/preprocess.stats
echo "The number of mRNAs with both GRO and DRS sites supported and are passed by filter: $num_consist" >> $saved_root/preprocess.stats
