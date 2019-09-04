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

preprocessed_root=$saved_root/preprocessed
processed_root=$saved_root/processed
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
gene_info_root=$script_root/sequence_annotation/gene_info

echo "Start of program"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root
mkdir -p $preprocessed_root

#Set parameter
genome_path=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
biomart_path=$root/raw_data/biomart_araport_11_gene_info_2018_11_27.csv
gro_1=$root/raw_data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
DRS_path=$root/raw_data/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv
peptide_path=$root/raw_data/Araport11_genes.201606.pep.fasta
id_convert_table_path=$preprocessed_root/id_convert.tsv
official_gff_path=$root/raw_data/Araport11_GFF3_genes_transposons.201606.gff
repaired_gff_path=$preprocessed_root/repaired_official.gff
processed_gff_path=$preprocessed_root/processed.gff
processed_bed_path=$preprocessed_root/processed.bed
echo "Step 2: Preprocess raw data"

if [ ! -e "$repaired_gff_path" ]; then
    python3 $gene_info_root/repair_gff.py -i $official_gff_path -o $repaired_gff_path -s $preprocessed_root
fi

if [ ! -e "$processed_gff_path" ]; then
    python3 $gene_info_root/preprocess_gff.py -i $repaired_gff_path -o $processed_gff_path
fi

if [ ! -e "$processed_bed_path" ]; then
    python3 $gene_info_root/gff2bed.py -i $processed_gff_path -o $processed_bed_path
fi
    
python3 $gene_info_root/preprocess_raw_data.py --output_root $preprocessed_root --bed_path $processed_bed_path \
--biomart_table_path $biomart_path --gro_1 $gro_1 --gro_2 $gro_2 --cs_path $DRS_path 

python3 $gene_info_root/get_external_UTR.py -b $preprocessed_root/valid_official.bed -s $preprocessed_root

echo "Step 3: Classify GRO and DRS data to belonging sites"
python3 $gene_info_root/classify_sites.py -o $preprocessed_root/valid_official.bed  -g $preprocessed_root/valid_gro.tsv -c $preprocessed_root/valid_cleavage_site.tsv -s $preprocessed_root -u $upstream_dist -d $downstream_dist \
-f $preprocessed_root/external_five_UTR.tsv -t $preprocessed_root/external_three_UTR.tsv > $preprocessed_root/classify.log

echo "Step 4: Get maximize signal sites data located on external UTR"
python3 $gene_info_root/consist_sites.py --ig $preprocessed_root/inner_gro_sites.tsv \
--ic $preprocessed_root/inner_cleavage_sites.tsv --lg $preprocessed_root/long_dist_gro_sites.tsv \
--lc $preprocessed_root/long_dist_cleavage_sites.tsv --tg $preprocessed_root/transcript_gro_sites.tsv \
--tc $preprocessed_root/transcript_cleavage_sites.tsv -s $preprocessed_root

echo "Step 5: Create coordiante data based on origin data and site data"
python3 $gene_info_root/create_coordinate_data.py -g $preprocessed_root/safe_gro_sites.tsv -c $preprocessed_root/safe_cleavage_sites.tsv -t $id_convert_table_path --single_start_end -o $preprocessed_root/coordinate_data.tsv

python3 $gene_info_root/create_coordinate_bed.py -i $preprocessed_root/valid_official.bed \
-c $preprocessed_root/coordinate_data.tsv -t $id_convert_table_path -o $preprocessed_root/coordinate_consist.bed

num_valid_official=$(wc -l < $preprocessed_root/valid_official.bed )
num_gro=$(sed "1d" $preprocessed_root/safe_gro_sites.tsv | wc -l)
num_drs=$(sed "1d" $preprocessed_root/safe_cleavage_sites.tsv | wc -l)
num_consist=$(wc -l < $preprocessed_root/coordinate_consist.bed )
   
echo "Selected mRNA count: $num_valid_official" > $preprocessed_root/preprocess.stats
echo "Matched GRO sites count: $num_gro" >> $preprocessed_root/preprocess.stats
echo "Matched DRS sites count: $num_drs" >> $preprocessed_root/preprocess.stats
echo "The number of mRNAs with both GRO and DRS sites supported and are passed by filter: $num_consist" >> $preprocessed_root/preprocess.stats

echo "Merge: $merge_overlapped"
if $merge_overlapped; then
    bash $bash_root/process_data.sh -u $upstream_dist -d $downstream_dist -g $genome_path -i $preprocessed_root/coordinate_consist.bed -o $processed_root -s $source_name -t $id_convert_table_path -b $processed_bed_path -m
else
    bash $bash_root/process_data.sh -u $upstream_dist -d $downstream_dist -g $genome_path -i $preprocessed_root/coordinate_consist.bed -o $processed_root -s $source_name -t $id_convert_table_path -b $processed_bed_path
fi

result_num=$(wc -l < $processed_root/result/selected_region.bed )

if  (( result_num ))  ;then

    python3 $gene_info_root/rename_bed.py -i $processed_root/result/selected_region.bed -p region -t $processed_root/result/region_rename_table_both_strand.tsv -o $processed_root/result/selected_region_both_strand.bed --use_id_as_id

    python3 $gene_info_root/redefine_coordinate.py -i $processed_root/result/rna.bed -t $processed_root/result/region_rename_table_both_strand.tsv -o $processed_root/result/rna_both_strand.bed --simple_mode
    
    bedtools getfasta -name -fi $genome_path -bed $processed_root/result/selected_region_both_strand.bed -fo $processed_root/result/selected_region_both_strand.fasta

    split_root=$processed_root/split
    mkdir -p $split_root

    python3 $gene_info_root/split.py --region_bed_path $processed_root/result/selected_region_both_strand.bed --region_rename_table_path $processed_root/result/region_rename_table_both_strand.tsv --fai_path $genome_path.fai --splitted_id_root $split_root

    for path in $(find $split_root/* -name '*.tsv');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        python3 $gene_info_root/get_subbed.py -i $processed_root/result/rna_both_strand.bed -d $split_root/$file_name.tsv \
        -o $split_root/$file_name.bed --query_column chr
        python3 $gene_info_root/get_GlimmerHMM_cds_file.py -i $split_root/$file_name.bed -o $split_root/$file_name.cds
        python3 $gene_info_root/bed2gff.py -i $split_root/$file_name.bed -o $split_root/$file_name.gff -t $id_convert_table_path
        python3 $gene_info_root/get_subfasta.py -i $processed_root/result/selected_region_both_strand.fasta -d $split_root/$file_name.tsv -o $split_root/$file_name.fasta
    done
    
    deep_learning_split_root=$processed_root/deep_learning_split
    mkdir -p $deep_learning_split_root

    python3 $gene_info_root/split.py --region_bed_path $processed_root/result/selected_region.bed --region_rename_table_path $processed_root/result/region_rename_table.tsv --fai_path $genome_path.fai --splitted_id_root $deep_learning_split_root

    for path in $(find $deep_learning_split_root/* -name '*.tsv');
    do
        file_name=$(basename $path)
        file_name="${file_name%.*}"
        python3 $gene_info_root/get_subbed.py -i $processed_root/result/rna.bed -d $deep_learning_split_root/$file_name.tsv \
        -o $deep_learning_split_root/$file_name.bed --query_column chr
        python3 $gene_info_root/get_subfasta.py -i $processed_root/result/selected_region.fasta -d $deep_learning_split_root/$file_name.tsv -o $deep_learning_split_root/$file_name.fasta
    done

    exit 0
else
    echo "The process_data.sh is failed"
    exit 1
fi
