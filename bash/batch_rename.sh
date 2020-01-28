#!/bin/bash
## function print usage
usage(){
 echo "Usage: Rename data's chromosome from single-strand id to double-strand id"
 echo "  Arguments:"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -t  <string>  Path of region_rename_table of single-strand to double-strand"
 echo "    -i  <string>  Path of processed folder"
 echo "    -o  <string>  Path of output folder"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash batch_rename.sh -i /home/io/split -t /home/io/preprocess/region_rename_table.tsv -o ./data/renamed -g genome.fasta"
 echo ""
}

while getopts g:t:i:o:h option
 do
  case "${option}"
  in
   g )genome_path=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   i )input_root=$OPTARG;;
   o )saved_root=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$genome_path" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi

if [ ! "$id_convert_table_path" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

if [ ! "$input_root" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

#Set parameter
bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
mkdir -p $saved_root

for path in $(find $input_root/* -name '*.bed');
do
    file_name=$(basename $path)
    file_name="${file_name%.*}"
    
    python3 $preprocess_main_root/rename_bed_chrom.py -i $input_root/$file_name.bed -t $ds_region_table_path -o $saved_root/$file_name.bed
    
    #python3 $preprocess_main_root/get_subfasta.py -i $output_region_fasta_path -d $saved_root/$file_name.txt -o $saved_root/$file_name.fasta

    python3 $preprocess_main_root/bed2gff.py -i $saved_root/$file_name.bed -o $saved_root/$file_name.gff -t $id_convert_table_path

    #samtools faidx $saved_root/$file_name.fasta

    #gene_num=$(wc -l < $saved_root/$file_name.txt ) 
    #echo "$file_name,$gene_num" >> $saved_root/count.csv
        
done


exit 0
