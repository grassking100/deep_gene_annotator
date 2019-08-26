#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating statistic data and fasta of bed data"
 echo "  Arguments:"
 echo "    -i  <string>  Path of bed"
 echo "    -f  <string>  Path of genome fasta"
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

while getopts i:f:p:o:s:t:c:d:a:h option
 do
  case "${option}"
  in
   i )bed_path=$OPTARG;;
   f )genome_path=$OPTARG;;
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

CDS_path=$output_root/CDS
donor_signal=$output_root/donor_signal
accept_signal=$output_root/accept_signal
tss_signal=$output_root/tss_signal
ca_signal=$output_root/ca_signal

if [ -e "$peptide_path.fai" ]; then
    rm $peptide_path.fai
fi
if [ -e "$CDS_path.fasta.fai" ]; then
    rm $CDS_path.fasta.fai
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
bash $bash_root/splice_donor_site.sh $bed_path 0 1 > $donor_signal.bed
bash $bash_root/splice_accept_site.sh $bed_path 1 0 > $accept_signal.bed
bash $bash_root/transcription_start_site.sh $bed_path 1 > $tss_signal.bed
bash $bash_root/cleavage_site.sh $bed_path 1 > $ca_signal.bed

for name in $tss_path $ca_path $splice_donor_path $splice_accept_path $donor_signal $accept_signal $tss_signal $ca_signal;
do
    bedtools getfasta -s -name -fi $genome_path -bed $name.bed -fo $name.fasta
done

python3 $src_root/get_CDS_bed.py -i $bed_path -o $CDS_path.bed

bedtools getfasta -s -fi $genome_path -name -split -bed $CDS_path.bed  -fo $CDS_path.fasta

awk -v 'RS=>' 'NR>1{
                        print ">" $1;
                        print(substr($2,1,3));
                   }'  $CDS_path.fasta > $start_codon_path.fasta
                    
awk -v 'RS=>' 'NR>1{
                        print ">" $1;
                        print(substr($2,length($2)-2,length($2)));
                   }'  $CDS_path.fasta > $stop_codon_path.fasta
                  
seq_stats () {
    local fasta_path=$1
    local stats_path=$2
    sed '/^>/ d' $fasta_path | tr a-z A-Z | sort | uniq -c | awk '{print $2 ": " $1}' > $stats_path
}

seq_stats $tss_signal.fasta $stats_root/tss_signal.stats
seq_stats $ca_signal.fasta $stats_root/ca_signal.stats
seq_stats $accept_signal.fasta $stats_root/accept_signal.stats
seq_stats $donor_signal.fasta  $stats_root/donor_signal.stats
seq_stats $start_codon_path.fasta $stats_root/start_codon.stats
seq_stats $stop_codon_path.fasta $stats_root/stop_codon.stats

sed '/^>/ d' $CDS_path.fasta | awk '{print length}' > $stats_root/CDS.length

printf "Longest length: %s\n" $(cat $stats_root/CDS.length | sort -rn | head -n 1) > $stats_root/CDS_length.stats
printf "Shortest length: %s\n" $(cat $stats_root/CDS.length | sort | head -n 1) >> $stats_root/CDS_length.stats


if [ -e "$subpeptide" ]; then
    sed '/^>/ d' $subpeptide | awk '{print substr($1,1,1)}' > $stats_root/first_aa.stats.temp
    seq_stats $stats_root/first_aa.stats.temp $stats_root/first_aa.stats
    rm $stats_root/first_aa.stats.temp
    sed '/^>/ d' $subpeptide | awk '{print length}' > $stats_root/peptide.length
    printf "Longest length: %s\n" $(cat $stats_root/peptide.length | sort -rn | head -n 1) > $stats_root/peptide_length.stats
    printf "Shortest length: %s\n" $(cat $stats_root/peptide.length | sort | head -n 1) >> $stats_root/peptide_length.stats
fi

python3 $src_root/bed2gff.py -i $bed_path -o $result_root/canonical.gff.temp

awk -F'\t' -v OFS="\t"  '{if($3=="exon"){print($5-$4+1)}}' $result_root/canonical.gff.temp > $stats_root/exon.length
awk -F'\t' -v OFS="\t"  '{if($3=="intron"){print($5-$4+1)}}' $result_root/canonical.gff.temp > $stats_root/intron.length
awk -F'\t' -v OFS="\t"  '{if($3=="UTR"){print($5-$4+1)}}' $result_root/canonical.gff.temp > $stats_root/utr.length
awk -F'\t' -v OFS="\t"  '{if($3=="CDS"){print($5-$4+1)}}' $result_root/canonical.gff.temp > $stats_root/cds.length

rm $result_root/canonical.gff.temp

exit 0