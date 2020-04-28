#!/bin/bash
## function print usage
usage(){
 echo "Usage: The process prepair Drosophila data"
 echo "  Arguments:"
 echo "    -r  <string>  Directory of Drosophila data"
 echo "    -o  <string>  Directory of output folder"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash drosophila_data_prepair.sh -r /home/io/Drosophila -o ./data/2019_07_12"
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

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess

#Set parameter
tss_path=$root/TSScall_tss/tss.bed
pac_path=$root/3_prime_seq_PAC/13059_2017_1358_MOESM3_ESM_D_melanogaster_dm6.tsv
official_gff_path=$root/gene/dmel-all-no-analysis-r5.57_gene.gff
genome_path=$root/genome/dmel-all-chromosome-r5.57.fasta
region_table_path=$saved_root/region_table.tsv
id_convert_table_path=$saved_root/id_convert.tsv
processed_gff_path=$saved_root/processed.gff3
processed_bed_path=$saved_root/processed.bed
consistent_gff_path=$saved_root/consistent_official.gff3
consistent_bed_path=$saved_root/consistent.bed
non_hypothetical_gene_id_path=$saved_root/non_hypothetical_gene_id.txt
echo "Step 1: Preprocess raw data"
#if [ ! -e "$genome_path.fai" ]; then
#    bash $bash_root/rename_fasta.sh $raw_genome_path > $genome_path
#    samtools faidx $genome_path
#fi
python3 $preprocess_main_root/preprocess_raw_data.py --output_root $saved_root --tss $tss_path --cleavage_site_tsv_path $pac_path
if [ ! -e "$processed_gff_path" ]; then
    python3 $preprocess_main_root/preprocess_gff.py -i $official_gff_path -o $processed_gff_path -v '1,2,3,4,5' \
    --non_hypothetical_gene_id_path $non_hypothetical_gene_id_path
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
