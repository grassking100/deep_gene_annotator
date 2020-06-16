#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating statistic data and fasta of bed data"
 echo "  Arguments:"
 echo "    -i  <string>  Path of GFF"
 echo "    -f  <string>  Path of genome fasta"
 echo "    -o  <string>  Directory to output"
 echo "    -r  <string>  Path of region table path"
 echo "    -s  <string>  Chromosome source (options: old_id, new_id)"
 echo "  Options:"
 echo "    -t  <int>     Radius of Transcription start sites                [default: 100]"
 echo "    -d  <int>     Radius of Donor sites                              [default: 100]"
 echo "    -a  <int>     Radius of Acceptor sites                           [default: 100]"
 echo "    -c  <int>     Radius of Cleavage sites                           [default: 100]"
 echo "    -h            Print help message and exit"
 echo "Example: bash gff_analysis.sh -i example.gff3 -f genome.fasta -o output -s old_id -r region_table.tsv"
 echo ""
}

while getopts i:f:o:t:c:d:a:r:s:h option
 do
  case "${option}"
  in
   i )gff_path=$OPTARG;;
   f )genome_path=$OPTARG;;
   o )output_root=$OPTARG;;
   t )tss_radius=$OPTARG;;
   c )cleavage_radius=$OPTARG;;
   d )donor_radius=$OPTARG;;
   a )acceptor_radius=$OPTARG;;
   r )region_table_path=$OPTARG;;
   s )chrom_source=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done


if [ ! "$gff_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi
if [ ! "$genome_path" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi
if [ ! "$output_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$region_table_path" ]; then
    echo "Missing option -r"
    usage
    exit 1
fi

if [ ! "$chrom_source" ]; then
    echo "Missing option -s"
    usage
    exit 1
fi

if [ ! "$tss_radius" ]; then
    tss_radius="100"
fi
if [ ! "$cleavage_radius" ]; then
    cleavage_radius="100"
fi
if [ ! "$donor_radius" ]; then
    donor_radius="100"
fi
if [ ! "$acceptor_radius" ]; then
    acceptor_radius="100"
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source_root=$bash_root/..
preprocess_main_root=$source_root/sequence_annotation/preprocess
visual_main_root=$source_root/sequence_annotation/visual
utils_root=$source_root/sequence_annotation/utils

bed_root=$output_root/bed
signal_stats_root=$output_root/signal_stats
fasta_root=$output_root/fasta
composition_root=$output_root/composition

#Create folder
mkdir -p $output_root
mkdir -p $signal_stats_root
mkdir -p $fasta_root
mkdir -p $composition_root

tss_path=tss_around_$tss_radius
cs_path=cleavage_site_around_$cleavage_radius
donor_path=donor_site_around_${donor_radius}
acceptor_path=acceptor_site_around_${acceptor_radius}
tss_signal_path=tss_signal
cleavage_site_signal_path=cleavage_site_signal
donor_signal_path=donor_site_signal
acceptor_signal_path=acceptor_site_signal

python3 $preprocess_main_root/signal_analysis.py -i $gff_path -o $bed_root -t $tss_radius -c $cleavage_radius -d $donor_radius -a $acceptor_radius

for path in $(find $bed_root/* -name '*.bed');
do
    file_name=$(basename $path)
    file_name="${file_name%.*}"
    bedtools getfasta -s -fi $genome_path -bed $bed_root/$file_name.bed -fo $fasta_root/$file_name.fasta
done

python3 $visual_main_root/plot_composition.py -i $fasta_root/$tss_path.fasta -s "-$tss_radius" --title "Nucleotide composition around TSS" -o $composition_root/$tss_path.png
python3 $visual_main_root/plot_composition.py -i $fasta_root/$cs_path.fasta -s "-$cleavage_radius" --title "Nucleotide composition around cleavage site" -o $composition_root/$cs_path.png
python3 $visual_main_root/plot_composition.py -i $fasta_root/$donor_path.fasta -s "-$donor_radius" --title "Nucleotide composition around splicing donor site" -o $composition_root/$donor_path.png
python3 $visual_main_root/plot_composition.py -i $fasta_root/$acceptor_path.fasta -s "-$acceptor_radius" --title "Nucleotide composition around splicing acceptor site" -o $composition_root/$acceptor_path.png

python3 $utils_root/motif_count.py -i $fasta_root/$tss_signal_path.fasta -o $signal_stats_root/tss_signal_stats.tsv
python3 $utils_root/motif_count.py -i $fasta_root/$cleavage_site_signal_path.fasta -o $signal_stats_root/cs_signal_stats.tsv
python3 $utils_root/motif_count.py -i $fasta_root/$donor_signal_path.fasta -o $signal_stats_root/donor_signal_stats.tsv
python3 $utils_root/motif_count.py -i $fasta_root/$acceptor_signal_path.fasta -o $signal_stats_root/acceptor_signal_stats.tsv

python3 $preprocess_main_root/gff_feature_stats.py -i $gff_path -r $region_table_path -c $chrom_source -o $output_root/feature_stats

#if [ ! -e "$output_root/length_gaussian.tsv" ]; then
python3 $preprocess_main_root/length_gaussian_modeling.py -i $gff_path -o $output_root/length_gaussian -n 2
#fi

echo "The gff_analysis is finish" > $output_root/gff_analysis.log

exit 0