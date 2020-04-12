#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline prepair annotation data"
 echo "  Arguments:"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -r  <string>  Directory of Arabidopsis thaliana data"
 echo "    -o  <string>  Directory of output folder"
 echo "    -s  <string>  Source name"
 echo "  Options:"
 echo "    -i  <bool>    Remove regions which inner signal                [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash arabidopsis_data_prepair.sh -u 10000 -d 10000 -r /home/io/Arabidopsis_thaliana -o ./data/2019_07_12 -s Arabidopsis_1"
 echo ""
}

while getopts u:d:r:o:s:ih option
 do
  case "${option}"
  in
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   r )root=$OPTARG;;
   o )saved_root=$OPTARG;;
   s )source_name=$OPTARG;;
   i )remove_transcript_external_UTR_conflict=true;;
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

if [ ! "$remove_transcript_external_UTR_conflict" ]; then
    remove_transcript_external_UTR_conflict=false
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
raw_stats_root=$saved_root/raw_stats
stats_root=$saved_root/stats
site_diff_root=$saved_root/site_diff

echo "Start of program"
#Create folder
echo "Step 1: Create folder"
mkdir -p $saved_root
mkdir -p $stats_root
mkdir -p $site_diff_root

#Set parameter
gro_1=$root/raw_data/tss_peak_SRR3647033_background_SRR3647034_2018_11_04.tsv 
gro_2=$root/raw_data/tss_peak_SRR3647033_background_SRR3647035_2018_11_04.tsv
pac_path="$root/raw_data/arabidopsis_thaliana_Wild_Type_Control_SRP187778_all_PAC.csv"
peptide_path=$root/raw_data/Araport11_genes.201606.pep.fasta
official_gff_path=$root/raw_data/Araport11_GFF3_genes_transposons.201606.gff
raw_genome_path=$root/raw_data/araport_11_Arabidopsis_thaliana_Col-0.fasta
genome_path=$saved_root/araport_11_Arabidopsis_thaliana_Col-0_rename.fasta
region_table_path=$saved_root/region_table.tsv
###
id_convert_table_path=$saved_root/id_convert.tsv
processed_gff_path=$saved_root/processed.gff3
processed_bed_path=$saved_root/processed.bed
consistent_gff_path=$saved_root/consistent_official.gff3
consistent_bed_path=$saved_root/consistent.bed
tss_path=$saved_root/tss.gff3
cleavage_site_path=$saved_root/cleavage_site.gff3
non_hypothetical_gene_id_path=$saved_root/non_hypothetical_gene_id.txt
echo "Step 2: Preprocess raw data"

if [ ! -e "$genome_path.fai" ]; then
    bash $bash_root/rename_fasta.sh $raw_genome_path > $genome_path
    samtools faidx $genome_path
fi

python3 $preprocess_main_root/process_GRO_and_PAC.py --output_root $saved_root --gro_1 $gro_1 --gro_2 $gro_2 --pac_path $pac_path

if [ ! -e "$processed_gff_path" ]; then
    python3 $preprocess_main_root/preprocess_gff.py -i $official_gff_path -o $processed_gff_path -v '1,2,3,4,5' --non_hypothetical_gene_id_path $non_hypothetical_gene_id_path
fi

if [ ! -e "$region_table_path" ]; then
    python3 $preprocess_main_root/create_region_table_by_fai.py -i $genome_path.fai -o $region_table_path
fi

if [ ! -e "$raw_stats_root/gff_analysis.log" ]; then
    bash $bash_root/gff_analysis.sh -i $processed_gff_path -f $genome_path -o $raw_stats_root -r $region_table_path -s chr
fi

if [ ! -e "$id_convert_table_path" ]; then
    python3 $preprocess_main_root/get_id_table.py -i $processed_gff_path -o $id_convert_table_path
fi

if [ ! -e "$processed_bed_path" ]; then
    python3 $preprocess_main_root/gff2bed.py -i $processed_gff_path -o $processed_bed_path
fi

if [ ! -e "$consistent_gff_path" ]; then
    python3 $preprocess_main_root/get_consistent_gff.py -i $processed_gff_path -s $saved_root -p 'official'
fi

if [ ! -e "$consistent_bed_path" ]; then
    python3 $preprocess_main_root/gff2bed.py -i $consistent_gff_path -o $consistent_bed_path
fi

if [ ! -e "$site_diff_root/site_diff.tsv" ]; then
    python3 $preprocess_main_root/calculate_distance_between_sites.py -b $consistent_bed_path -t $tss_path -c $cleavage_site_path -s $site_diff_root
fi

python3 $preprocess_main_root/get_external_UTR.py -b $consistent_bed_path -s $saved_root

echo "Step 3: Classify TSS and CS data to belonging site"
python3 $preprocess_main_root/classify_site.py -o $consistent_bed_path  -g $tss_path -c $cleavage_site_path -s $saved_root -u $upstream_dist -d $downstream_dist \
-f $saved_root/external_five_UTR.bed -t $saved_root/external_three_UTR.bed

echo "Step 4: Get maximize signal site data located on external UTR"
command="$preprocess_main_root/consist_site.py --external_five_UTR_tss_path $saved_root/external_five_UTR_tss.gff3 \
--external_three_UTR_cs_path $saved_root/external_three_UTR_cleavage_site.gff3 --long_dist_tss_path $saved_root/long_dist_tss.gff3 \
--long_dist_cs_path $saved_root/long_dist_cleavage_site.gff3 -s $saved_root --transcript_tss_path $saved_root/transcript_tss.gff3 \
--transcript_cs_path $saved_root/transcript_cleavage_site.gff3"

if $remove_transcript_external_UTR_conflict; then
    command="${command} --remove_transcript_external_UTR_conflict"
fi
    
python3 $command

echo "Step 5: Create coordiante data based on origin data and site data"
python3 $preprocess_main_root/create_coordinate_data.py -g $saved_root/safe_tss.gff3 -c $saved_root/safe_cs.gff3 -t $id_convert_table_path --single_start_end -o $saved_root/coordinate.gff3

python3 $preprocess_main_root/create_coordinate_bed.py -i $consistent_bed_path \
-c $saved_root/coordinate.gff3 -t $id_convert_table_path -o $saved_root/coordinate_consist.bed

python3 $preprocess_main_root/coordinate_compare.py -r $consistent_bed_path -c $saved_root/coordinate_consist.bed -o $saved_root/coordinate_compared.gff3

num_consistent=$(wc -l < $consistent_bed_path )
num_raw_tss=$(sed "1d"  $saved_root/tss.gff3 | wc -l )
num_raw_cs=$(sed "1d" $saved_root/cleavage_site.gff3 | wc -l)
num_external_five_UTR_tss=$(sed "1d" $saved_root/external_five_UTR_tss.gff3 | wc -l)
num_external_three_UTR_cleavage_site=$(sed "1d" $saved_root/external_three_UTR_cleavage_site.gff3 | wc -l)
num_safe_tss=$(sed "1d" $saved_root/safe_tss.gff3 | wc -l)
num_safe_cs=$(sed "1d" $saved_root/safe_cs.gff3 | wc -l)
num_consist=$(wc -l < $saved_root/coordinate_consist.bed )
   
echo "Selected mRNA count: $num_consistent" > $saved_root/preprocess.stats
echo "Evidence TSS count: $num_raw_tss" >> $saved_root/preprocess.stats
echo "Evidence CS count: $num_raw_cs" >> $saved_root/preprocess.stats
echo "Evidence TSS located at external 5' UTR: $num_external_five_UTR_tss" >> $saved_root/preprocess.stats
echo "Evidence CS located at external 3' UTR count: $num_external_three_UTR_cleavage_site" >> $saved_root/preprocess.stats
echo "Evidence TSS located at external 5' UTR and is most significant signal: $num_safe_tss" >> $saved_root/preprocess.stats
echo "Evidence CS located at external 3' UTR and is most significant signal: $num_safe_cs" >> $saved_root/preprocess.stats
echo "The number of mRNAs with both TSSs and CSs site supported and are passed by filter: $num_consist" >> $saved_root/preprocess.stats

if [ ! -e "$stats_root/gff_analysis.log" ]; then
    bash  $bash_root/gff_analysis.sh -i $processed_gff_path -f $genome_path -o $stats_root -r $region_table_path -s chr
fi
