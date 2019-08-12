#!/bin/bash
folder_name="termite_data_2019_07_23"
root="/home/io/long_read/termite"
saved_root=$root/$folder_name
coordinate_bed=$saved_root/gmap/gmap_align.bed
genome_file=$root/raw_data/genome/termite_genome2.fasta
bash sequence_annotation/bash/process_data.sh -u 2000 -w 2000 -g $genome_file -i $coordinate_bed -o $saved_root -s $folder_name -f true
cp -t $saved_root ${BASH_SOURCE[0]}
