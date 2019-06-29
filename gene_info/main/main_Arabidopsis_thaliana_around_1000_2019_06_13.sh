#!/bin/bash
folder_name=2019_06_13
upstream_dist=1000
downstream_dist=1000
TSS_radius=100
donor_radius=100
accept_radius=100
cleavage_radius=100
root="./io/Arabidopsis_thaliana"
saved_root=$root/data/$folder_name
result_root=$saved_root/result
separate_root=$saved_root/separate
fasta_root=$saved_root/fasta
script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/..
src_root=$script_root/src
echo "Start of program"
mkdir -p $saved_root
mkdir -p $result_root
mkdir -p $separate_root
mkdir -p $fasta_root
cp -t $saved_root ${BASH_SOURCE[0]}
#Set parameter
fai=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta.fai
genome_file=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
bed_target_path=$root/raw_data/Araport11_GFF3_genes_transposons.201606_repair_2019_05_11.bed
biomart_path=$root/raw_data/biomart_araport_11_gene_info_2018_11_27.csv
gro_1=$root/raw_data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
drs=$root/raw_data/NIHMS48846-supplement-2_S10_DRS_peaks_in_coding_genes_private.csv
peptide_file=$root/raw_data/Araport11_genes.201606.pep.fasta
id_convert=$saved_root/id_convert.tsv
result_merged=result_upstream_${upstream_dist}_downstream_${downstream_dist}_merged
#Preprocess
python3 $src_root/preprocess_raw_data.py --saved_root $saved_root --bed_path $bed_target_path \
--biomart_path $biomart_path --gro_1 $gro_1 --gro_2 $gro_2 --cs_path $drs

python3 $src_root/get_most_UTR.py -b $saved_root/valid_official_coding.bed -s $saved_root

python3 $src_root/classify_sites.py -o $saved_root/valid_official_coding.bed  -g $saved_root/valid_gro.tsv -c $saved_root/valid_cleavage_site.tsv -s $saved_root -u $upstream_dist -d $downstream_dist \
-f $saved_root/most_five_UTR.tsv -t $saved_root/most_three_UTR.tsv 

python3 $src_root/consist_sites.py --ig $saved_root/inner_gro_sites.tsv \
--ic $saved_root/inner_cleavage_sites.tsv --lg $saved_root/long_dist_gro_sites.tsv \
--lc $saved_root/long_dist_cleavage_sites.tsv --tg $saved_root/transcript_gro_sites.tsv \
--tc $saved_root/transcript_cleavage_sites.tsv -s $saved_root

#Write to coordinate_consist.bed
python3 $src_root/create_coordinate_data.py -s $saved_root -g $saved_root/safe_merged_gro_sites.tsv -c $saved_root/safe_merged_cleavage_sites.tsv -i $id_convert
python3 $src_root/create_coordinate_bed.py -s $saved_root -c $saved_root/consist_data.tsv \
-o $saved_root/valid_official_coding.bed -i $id_convert
#Remove overlap gene
python3 $src_root/nonoverlap_filter.py -c $saved_root/coordinate_consist.bed -i $id_convert -s $separate_root
#Remove overlap gene based on certain distance
python3 $src_root/recurrent_cleaner.py -r $saved_root/valid_official_coding.bed -c $separate_root/nonoverlap.bed -f $fai -u $upstream_dist -d $downstream_dist -s $separate_root -i $id_convert

cp $separate_root/recurrent_cleaned.bed $fasta_root/result.bed
python3 $src_root/get_around_fasta.py -b $fasta_root/result.bed -t $TSS_radius -d $donor_radius -a $accept_radius -c $cleavage_radius -s $fasta_root -g $genome_file -p $peptide_file

cp $separate_root/recurrent_cleaned.bed $result_root/result.bed
bash $script_root/bash/get_region.sh $result_root/result.bed $fai $upstream_dist $downstream_dist 

python3 $src_root/rename_bed.py -b $result_root/$result_merged.bed -p seq -s $result_root -r selected_region.bed
rm $result_root/$result_merged.bed
bedtools getfasta -s -name -fi $genome_file -bed $result_root/selected_region.bed -fo $fasta_root/selected_region.fasta
python3 $src_root/create_ann_region.py -m $result_root/result.bed -r $result_root/selected_region.bed \
-f $fai -s $folder_name -o $result_root/selected_region.h5 --saved_root $result_root
echo "End of program"