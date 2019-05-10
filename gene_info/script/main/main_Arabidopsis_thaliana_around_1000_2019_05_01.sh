#!/bin/bash
folder_name=2019_05_01
upstream_dist=500
downstream_dist=500
TSS_radius=100
donor_radius=100
accept_radius=100
cleavage_radius=100
root="./io/Arabidopsis_thaliana/data"
saved_root=$root/$folder_name
result_path=$saved_root/result
separate_path=$saved_root/separate
fasta_root=$saved_root/fasta
echo "Start of program"
#rm -r $saved_root
mkdir -p $saved_root
mkdir -p $result_path
mkdir -p $separate_path
mkdir -p $fasta_root
cp -t $saved_root ${BASH_SOURCE[0]}
#Set parameter
fai=$root/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta.fai
genome_file=$root/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bed_target_path=$root/Araport11_GFF3_genes_transposons.201606_coding_repair_2019_04_07.bed
id_convert=$saved_root/id_convert.tsv
biomart_path=$root/biomart_araport_11_gene_info_2018_11_27.csv
gro_1=$root/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
drs=$root/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv

result_merged=result_upstream_${upstream_dist}_downstream_${downstream_dist}_merged
#Preprocess
python3 $script_root/python/preprocess_raw_data.py --saved_root $saved_root --bed_path $bed_target_path \
--biomart_path $biomart_path --gro_1 $gro_1 --gro_2 $gro_2 --cs_path $drs

python3 $script_root/python/get_most_UTR.py -b $saved_root/valid_official_coding.bed -s $saved_root

python3 $script_root/python/classify_sites.py -o $saved_root/valid_official_coding.bed  -g $saved_root/valid_gro.tsv -c $saved_root/valid_cleavage_site.tsv -s $saved_root -u $upstream_dist -d $downstream_dist \
-f $saved_root/most_five_UTR.tsv -t $saved_root/most_three_UTR.tsv 

python3 $script_root/python/consist_sites.py --ig $saved_root/inner_gro_sites.tsv \
--ic $saved_root/inner_cleavage_sites.tsv --lg $saved_root/long_dist_gro_sites.tsv \
--lc $saved_root/long_dist_cleavage_sites.tsv --tg $saved_root/transcript_gro_sites.tsv \
--tc $saved_root/transcript_cleavage_sites.tsv -s $saved_root

#Write to coordinate_consist.bed
python3 $script_root/python/create_coordinate_data.py -s $saved_root -g $saved_root/safe_merged_gro_sites.tsv -c $saved_root/safe_merged_cleavage_sites.tsv -i $id_convert
python3 $script_root/python/create_coordinate_bed.py -s $saved_root -c $saved_root/consist_data.tsv \
-o $saved_root/valid_official_coding.bed -i $id_convert
#Remove overlap gene
python3 $script_root/python/nonoverlap_filter.py -c $saved_root/coordinate_consist.bed -i $id_convert -s $separate_path
#Remove overlap gene based on certain distance
python3 $script_root/python/recurrent_cleaner.py -r $saved_root/valid_official_coding.bed -c $separate_path/nonoverlap.bed -f $fai \
-u $upstream_dist -d $downstream_dist -s $separate_path -i $id_convert

cp $separate_path/recurrent_cleaned.bed $fasta_root/result.bed
cp $separate_path/recurrent_cleaned.bed $result_path/result.bed
python3 $script_root/python/get_around_fasta.py -b $fasta_root/result.bed -u $upstream_dist \
-o $downstream_dist -t $TSS_radius -d $donor_radius -a $accept_radius -c $cleavage_radius -s $fasta_root -g $genome_file
cp $fasta_root/$result_merged.bed $result_path/$result_merged.bed

python3 $script_root/python/create_ann_region.py -m $result_path/result.bed -r $result_path/$result_merged.bed \
-f $fai -s $folder_name -o $result_path/$result_merged.h5 --saved_root $result_path
echo "End of program"