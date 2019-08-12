#!/bin/bash
folder_name=filter_orf_2019_07_31
root="/home/io/Arabidopsis_thaliana"
saved_root=$root/data/$folder_name
bash sequence_annotation/bash/arabidopsis_main.sh -u 2000 -w 2000 -r $root -o $saved_root -s $folder_name -f true
cp -t $saved_root ${BASH_SOURCE[0]}