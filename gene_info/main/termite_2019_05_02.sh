#!/bin/bash
folder_name="isoseq.ccs.polished.hq.fasta_align_to_termite_g1_2019_05_02"
root="io/long_read/termite"
script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
saved_root=${root}/${folder_name}
coordinate_bed=${saved_root}/transdecoder_complete_cds_align_to_termite_g1_2018_9_7_Ching_Tien_Wang
separate_path=${saved_root}/separate
fasta_root=${saved_root}/fasta
want_bed=${separate_path}/want
unwant_bed=${separate_path}/unwant
result_path=${saved_root}/result
fai=${root}/raw_data/genome/termite_g1.fasta.fai
genome_file=${root}/raw_data/genome/termite_g1.fasta
upstream_dist=200
downstream_dist=200
TSS_radius=100
donor_radius=100
accept_radius=100
cleavage_radius=100
echo "Start of program"
mkdir ${result_path}
mkdir ${separate_path}
mkdir ${saved_root}
mkdir ${fasta_root}
cp -t $saved_root ${BASH_SOURCE[0]}
result_merged=result_upstream_${upstream_dist}_downstream_${downstream_dist}_merged
python3 $script_root/python/create_id_table_by_coord.py -b $coordinate_bed.bed -s $saved_root
python3 $script_root/python/nonoverlap_filter.py -c $coordinate_bed.bed -s $separate_path -i $saved_root/id_table.tsv
#Remove overlap gene based on certain distance
python3 $script_root/python/recurrent_cleaner.py -r $coordinate_bed.bed -c $separate_path/nonoverlap.bed -f $fai -u $upstream_dist -d $downstream_dist -s $separate_path -i $saved_root/id_table.tsv

cp $separate_path/recurrent_cleaned.bed $fasta_root/result.bed
cp $separate_path/recurrent_cleaned.bed $result_path/result.bed

python3 $script_root/python/get_around_fasta.py -b $fasta_root/result.bed -u $upstream_dist \
-o $downstream_dist -t $TSS_radius -d $donor_radius -a $accept_radius -c $cleavage_radius -s $fasta_root -g $genome_file

cp $fasta_root/$result_merged.bed $result_path/$result_merged.bed

python3 $script_root/python/create_ann_region.py -m $result_path/result.bed -r $result_path/$result_merged.bed \
-f $fai -s $folder_name -o $result_path/$result_merged.h5 --saved_root $result_path

echo "End of program"