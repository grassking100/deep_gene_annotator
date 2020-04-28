#!/bin/bash
## function print usage
usage(){
 echo "Usage: The process prepair arabidopsis data"
 echo "  Arguments:"
 echo "    -r  <string>  Directory of Arabidopsis thaliana data"
 echo "    -o  <string>  Directory of output folder"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash prepair_data.sh -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12"
 echo ""
}

while getopts r:o:h option
 do
  case "${option}"
  in
   r )root=$OPTARG;;
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
mkdir -p $saved_root
kwargs_path=$saved_root/arabidopsis_data_prepair_kwargs.csv
echo "name,value" > $kwargs_path
echo "root,$root" >> $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path

arabidopsis_util_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$arabidopsis_util_root/../..
preprocess_main_root=$script_root/sequence_annotation/preprocess
bash_root=$script_root/bash
#Set parameter
gro_1=$root/raw_data/homer_tss/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/homer_tss/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
pac_path="$root/raw_data/PlantAPAdb_pac/arabidopsis_thaliana.Seedling_Control_(SRP089899).all.PAC.bed"
official_gff_path=$root/raw_data/gene/Araport11_GFF3_genes_transposons.201606.gff
raw_genome_path=$root/raw_data/genome/araport_11_Arabidopsis_thaliana_Col-0.fasta
genome_path=$saved_root/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
region_table_path=$saved_root/region_table.tsv
id_convert_table_path=$saved_root/id_convert.tsv
processed_gff_path=$saved_root/processed.gff3
processed_bed_path=$saved_root/processed.bed
consistent_gff_path=$saved_root/consistent_official.gff3
consistent_bed_path=$saved_root/consistent.bed

echo "Step 1: Preprocess raw data"
if [ ! -e "$genome_path.fai" ]; then
    bash $bash_root/rename_fasta.sh $raw_genome_path > $genome_path
    samtools faidx $genome_path
fi
python3 $arabidopsis_util_root/preprocess_raw_data.py --output_root $saved_root --gro_1 $gro_1 \
--gro_2 $gro_2 --pac_path $pac_path
if [ ! -e "$processed_gff_path" ]; then
    python3 $arabidopsis_util_root/preprocess_gff.py -i $official_gff_path -o $processed_gff_path \
    -v '1,2,3,4,5'
fi
if [ ! -e "$region_table_path" ]; then
    python3 $preprocess_main_root/create_region_table_by_fai.py -i $genome_path.fai -o $region_table_path
fi
if [ ! -e "$id_convert_table_path" ]; then
    python3 $preprocess_main_root/get_id_table.py -i $processed_gff_path -o $id_convert_table_path
fi
if [ ! -e "$processed_bed_path" ]; then
    python3 $preprocess_main_root/gff2bed.py -i $processed_gff_path -o $processed_bed_path
fi

echo "Step 2: Get consistence GFF file"
if [ ! -e "$consistent_gff_path" ]; then
    python3 $preprocess_main_root/get_consistent_gff.py -i $processed_gff_path -s $saved_root -p 'official'
fi

if [ ! -e "$consistent_bed_path" ]; then
    python3 $preprocess_main_root/gff2bed.py -i $consistent_gff_path -o $consistent_bed_path
fi

