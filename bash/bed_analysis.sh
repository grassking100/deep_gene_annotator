#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating statistic data and fasta of bed data"
 echo "  Arguments:"
 echo "    -i  <string>  Path of bed"
 echo "    -g  <string>  Path of genome fasta"
 echo "    -o  <string>  Directory of output"
 echo "    -s  <string>  Directory of statistic data"
 echo "  Options:"
 echo "    -t  <int>     Radius of Transcription start sites                [default: 100]"
 echo "    -d  <int>     Radius of Donor sites    [default: 100]"
 echo "    -a  <int>     Radius of Accept sites   [default: 100]"
 echo "    -c  <int>     Radius of Cleavage sites                           [default: 100]"
 echo "    -p  <string>  Path of peptide fasta"
 echo "    -h            Print help message and exit"
 echo "Example: bash bed_analysis.sh -i example.bed -g genome.fasta -p peptide.fasta -o output -o stats"
 echo ""
}

while getopts i:g:p:o:s:t:c:d:a:h option
 do
  case "${option}"
  in
   i )bed_path=$OPTARG;;
   g )genome_path=$OPTARG;;
   p )peptide_path=$OPTARG;;
   o )output_root=$OPTARG;;
   s )stats_root=$OPTARG;;
   t )tss_radius=$OPTARG;;
   c )cleavage_radius=$OPTARG;;
   d )donor_radius=$OPTARG;;
   a )accept_radius=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$bed_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi
if [ ! "$genome_path" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi
if [ ! "$output_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi
if [ ! "$stats_root" ]; then
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
if [ ! "$accept_radius" ]; then
    accept_radius="100"
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
sa_root=$bash_root/..
src_root=$sa_root/sequence_annotation/gene_info

#Create folder
mkdir -p $output_root
mkdir -p $stats_root

tss_path=$output_root/tss_around_$tss_radius
ca_path=$output_root/ca_around_$cleavage_radius
splice_donor_path=$output_root/splice_donor_around_${donor_radius}
splice_accept_path=$output_root/splice_accept_around_${accept_radius}
start_codon_path=$output_root/start_codon
stop_codon_path=$output_root/stop_codon

cDNA_path=$output_root/cDNA
donor_signal=$output_root/donor_signal
accept_signal=$output_root/accept_signal
tss_signal=$output_root/tss_signal
ca_signal=$output_root/ca_signal

if [ -e "$peptide_path.fai" ]; then
    rm $peptide_path.fai
fi
if [ -e "$cDNA_path.fasta.fai" ]; then
    rm $cDNA_path.fasta.fai
fi

if [ -e "$peptide_path" ]; then
    subpeptide=$output_root/peptide.fasta
    awk -F'\t' -v OFS="\t"  '{print($4)}' $bed_path > $output_root.id
    python3 $src_root/get_subfasta.py -i $peptide_path -d $output_root.id -o $subpeptide
fi

bash $bash_root/transcription_start_site.sh $bed_path $tss_radius > $tss_path.bed
bash $bash_root/cleavage_site.sh $bed_path $cleavage_radius > $ca_path.bed
bash $bash_root/splice_donor_site.sh $bed_path ${donor_radius} ${donor_radius} > $splice_donor_path.bed
bash $bash_root/splice_accept_site.sh $bed_path ${accept_radius} ${accept_radius} > $splice_accept_path.bed
bash $bash_root/start_codon.sh $bed_path > $start_codon_path.bed
bash $bash_root/stop_codon.sh $bed_path > $stop_codon_path.bed
bash $bash_root/splice_donor_site.sh $bed_path 0 1 > $donor_signal.bed
bash $bash_root/splice_accept_site.sh $bed_path 1 0 > $accept_signal.bed
bash $bash_root/transcription_start_site.sh $bed_path 1 > $tss_signal.bed
bash $bash_root/cleavage_site.sh $bed_path 1 > $ca_signal.bed

for name in $tss_path $ca_path $splice_donor_path $splice_accept_path $donor_signal $accept_signal $tss_signal $ca_signal;
do
    python3 $src_root/simply_coord.py -i $name.bed -o $name.bed
    bedtools getfasta -s -fi $genome_path -bed $name.bed -fo $name.fasta
done

bedtools getfasta -s -fi $genome_path -name -split -bed $bed_path  -fo $cDNA_path.fasta
for name in $start_codon_path $stop_codon_path;
do
    python3 $src_root/simply_coord.py -i $name.bed -o $name.bed
    bedtools getfasta -name -fi $cDNA_path.fasta -bed $name.bed -fo $name.fasta
done

sed '/^>/ d' $start_codon_path.fasta | sort | uniq -c | awk '{print $2 ": " $1}' > $stats_root/start_codon.stats
sed '/^>/ d' $stop_codon_path.fasta | sort | uniq -c | awk '{print $2 ": " $1}' > $stats_root/stop_codon.stats
sed '/^>/ d' $cDNA_path.fasta | awk '{print length}' > $stats_root/cDNA.length
sed '/^>/ d' $accept_signal.fasta | sort | uniq -c | awk '{print $2 ": " $1}' > $stats_root/accept_signal.stats
sed '/^>/ d' $donor_signal.fasta | sort | uniq -c | awk '{print $2 ": " $1}' > $stats_root/donor_signal.stats
sed '/^>/ d' $tss_signal.fasta | sort | uniq -c | awk '{print $2 ": " $1}' > $stats_root/tss_signal.stats
sed '/^>/ d' $ca_signal.fasta | sort | uniq -c | awk '{print $2 ": " $1}' > $stats_root/ca_signal.stats

printf "Longest length: %s\n" $(cat $stats_root/cDNA.length | sort -rn | head -n 1) > $stats_root/cDNA_length.stats
printf "Shortest length: %s\n" $(cat $stats_root/cDNA.length | sort | head -n 1) >> $stats_root/cDNA_length.stats


if [ -e "$subpeptide" ]; then
    sed '/^>/ d' $subpeptide | awk '{print substr($1,1,1)}'| sort | uniq -c | awk '{print $2 ": " $1}' > $stats_root/first_aa.stats
    sed '/^>/ d' $subpeptide | awk '{print length}' > $stats_root/peptide.length
    printf "Longest length: %s\n" $(cat $stats_root/peptide.length | sort -rn | head -n 1) > $stats_root/peptide_length.stats
    printf "Shortest length: %s\n" $(cat $stats_root/peptide.length | sort | head -n 1) >> $stats_root/peptide_length.stats
fi

exit 0