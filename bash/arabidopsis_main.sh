#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating annotation data"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -r  <string>  Directory of Arabidopsis thaliana data"
 echo "    -o  <string>  Directory of output folder"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -m  <bool>    Merge regions which are overlapped                 [default: false]"
 
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_main.sh -u 10000 -d 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:r:o:s:mh option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   r )root=$OPTARG;;
   o )saved_root=$OPTARG;;
   s )source_name=$OPTARG;;
  
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


if [ ! "$merge_overlapped" ]; then
    merge_overlapped=false
fi

processed_root=$saved_root/processed
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
biomart_path=$root/raw_data/biomart_araport_11_gene_info_2018_11_27.csv
gro_1=$root/raw_data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
DRS_path=$root/raw_data/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv
peptide_path=$root/raw_data/Araport11_genes.201606.pep.fasta
id_convert_table_path=$processed_root/id_convert.tsv
official_gff_path=$root/raw_data/Araport11_GFF3_genes_transposons.201606.gff
repaired_gff_path=$processed_root/repaired_official.gff
processed_gff_path=$processed_root/processed.gff
processed_bed_path=$processed_root/processed.bed
echo "Step 2: Preprocess raw data"

if [ ! -e "$repaired_gff_path" ]; then
    python3 $gene_info_root/repair_gff.py -i $official_gff_path -o $repaired_gff_path -s $processed_root
fi

if [ ! -e "$processed_gff_path" ]; then
    python3 $gene_info_root/preprocess_gff.py -i $repaired_gff_path -o $processed_gff_path
fi

if [ ! -e "$processed_bed_path" ]; then
    python3 $gene_info_root/gff2bed.py -i $processed_gff_path -o $processed_bed_path
fi
    
python3 $gene_info_root/preprocess_raw_data.py --output_root $processed_root --bed_path $processed_bed_path \
--biomart_table_path $biomart_path --gro_1 $gro_1 --gro_2 $gro_2 --cs_path $DRS_path 

python3 $gene_info_root/get_external_UTR.py -b $processed_root/valid_official.bed -s $processed_root

echo "Step 3: Classify GRO and DRS data to belonging sites"
python3 $gene_info_root/classify_sites.py -o $processed_root/valid_official.bed  -g $processed_root/valid_gro.tsv -c $processed_root/valid_cleavage_site.tsv -s $processed_root -u $upstream_dist -d $downstream_dist \
-f $processed_root/external_five_UTR.tsv -t $processed_root/external_three_UTR.tsv > $processed_root/classify.log

echo "Step 4: Get maximize signal sites data located on external UTR"
python3 $gene_info_root/consist_sites.py --ig $processed_root/inner_gro_sites.tsv \
--ic $processed_root/inner_cleavage_sites.tsv --lg $processed_root/long_dist_gro_sites.tsv \
--lc $processed_root/long_dist_cleavage_sites.tsv --tg $processed_root/transcript_gro_sites.tsv \
--tc $processed_root/transcript_cleavage_sites.tsv -s $processed_root

echo "Step 5: Create coordiante data based on origin data and site data"
python3 $gene_info_root/create_coordinate_data.py -g $processed_root/safe_gro_sites.tsv -c $processed_root/safe_cleavage_sites.tsv -t $id_convert_table_path --single_start_end -o $processed_root/coordinate_data.tsv

python3 $gene_info_root/create_coordinate_bed.py -i $processed_root/valid_official.bed \
-c $processed_root/coordinate_data.tsv -t $id_convert_table_path -o $processed_root/coordinate_consist.bed

num_valid_official=$(wc -l < $processed_root/valid_official.bed )
num_gro=$(sed "1d" $processed_root/safe_gro_sites.tsv | wc -l)
num_drs=$(sed "1d" $processed_root/safe_cleavage_sites.tsv | wc -l)
num_consist=$(wc -l < $processed_root/coordinate_consist.bed )
   
echo "Selected mRNA count: $num_valid_official" > $processed_root/preprocess.stats
echo "Matched GRO sites count: $num_gro" >> $processed_root/preprocess.stats
echo "Matched DRS sites count: $num_drs" >> $processed_root/preprocess.stats
echo "The number of mRNAs with both GRO and DRS sites supported and are passed by filter: $num_consist" >> $processed_root/preprocess.stats

if $merge_overlapped; then
    bash $bash_root/process_data.sh -u $upstream_dist -d $downstream_dist -g $genome_path -i $processed_root/coordinate_consist.bed -o $saved_root -s $source_name -t $id_convert_table_path -b $processed_bed_path -m
else
    bash $bash_root/process_data.sh -u $upstream_dist -d $downstream_dist -g $genome_path -i $processed_root/coordinate_consist.bed -o $saved_root -s $source_name -t $id_convert_table_path -b $processed_bed_path
fi

result_num=$(wc -l < $saved_root/result/selected_region.bed )

if  (( result_num ))  ;then

    split_root=$saved_root/result/split
    mkdir -p $split_root

    python3 $gene_info_root/split.py --region_bed_path $saved_root/result/selected_region.bed --region_rename_table_path $saved_root/result/region_rename_table.tsv --fai_path $genome_path.fai --splitted_id_root $split_root

    for path in $(find $saved_root/result/split/* -name '*.tsv');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        python3 $gene_info_root/get_subbed.py -i $saved_root/result/cleaned.bed -d $split_root/$file_name.tsv \
        -o $split_root/$file_name.bed --query_column chr
        python3 $gene_info_root/get_GlimmerHMM_cds_file.py -i $split_root/$file_name.bed -o $split_root/$file_name.cds
        python3 $gene_info_root/bed2gff.py -i $split_root/$file_name.bed -o $split_root/$file_name.gff -t $id_convert_table_path
        python3 $gene_info_root/get_subfasta.py -i $saved_root/result/selected_region.fasta -d $split_root/$file_name.tsv -o $split_root/$file_name.fasta
    done
    exit 0
else
    echo "The process_data.sh is failed"
    exit 1
fi

