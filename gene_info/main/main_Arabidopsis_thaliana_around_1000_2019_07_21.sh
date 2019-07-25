#!/bin/bash
folder_name=2019_07_21
root="/home/io/Arabidopsis_thaliana"
saved_root=$root/data/$folder_name
bash sequence_annotation/gene_info/bash/arabidopsis_main.sh -u 2000 -w 2000 -r $root -o $saved_root -s $folder_name
cp -t $saved_root ${BASH_SOURCE[0]}