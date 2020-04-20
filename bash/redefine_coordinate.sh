#!/bin/bash
## function print usage
usage(){
 echo "Usage: The process would redifine the coordinate of transcript based on TSS and CS data"
 echo "  Arguments:"
 echo "    -b  <string>  Transcript bed file"
 echo "    -t  <string>  TSS gff path"
 echo "    -c  <string>  Cleavage site gff path"
 echo "    -i  <string>  Transcript and gene id conversion table"
 echo "    -u  <int>     Upstream distance"
 echo "    -d  <int>     Downstream distance"
 echo "    -o  <string>  Directory of output folder"
 echo "  Options:"
 echo "    -e  <bool>    Remove region which the strongest signal in transcript but not in external UTR    [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash redefine_coordinate.sh -b transcript.bed -t tss.gff -c cs.gff  -u 10000 -d 10000 -i id_table.tsv  -o ./data/2019_07_12 -x"
 echo ""
}

while getopts b:t:c:i:u:d:o:eh option
 do
  case "${option}"
  in
    b )transcript_bed_path=$OPTARG;;
    t )tss_path=$OPTARG;;
    c )cleavage_site_path=$OPTARG;;
    i )id_convert_table_path=$OPTARG;;
    u )upstream_dist=$OPTARG;;
    d )downstream_dist=$OPTARG;;
    o )saved_root=$OPTARG;;
    e )remove_transcript_external_UTR_conflict=true;;
    h )usage; exit 1;;
    : )echo "Option $OPTARG requires an argument"
       usage; exit 1;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1;;
 esac
done

if [ ! "$transcript_bed_path" ]; then
    echo "Missing option -b"
    usage
    exit 1
fi

if [ ! "$tss_path" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

if [ ! "$cleavage_site_path" ]; then
    echo "Missing option -c"
    usage
    exit 1
fi

if [ ! "$id_convert_table_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi

if [ ! "$upstream_dist" ]; then
    echo "Missing option -u"
    usage
    exit 1
fi

if [ ! "$downstream_dist" ]; then
    echo "Missing option -d"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$remove_transcript_external_UTR_conflict" ]; then
    remove_transcript_external_UTR_conflict=false
fi

site_diff_root=$saved_root/site_diff
mkdir -p $saved_root
mkdir -p $site_diff_root

kwargs_path=$saved_root/coordinate_redefined_kwargs.csv
echo "name,value" > $kwargs_path
echo "transcript_bed_path,$transcript_bed_path" >> $kwargs_path
echo "tss_path,$tss_path" >> $kwargs_path
echo "cleavage_site_path,$cleavage_site_path" >> $kwargs_path
echo "upstream_dist,$upstream_dist" >> $kwargs_path
echo "downstream_dist,$downstream_dist" >> $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path
echo "remove_transcript_external_UTR_conflict,$remove_transcript_external_UTR_conflict" >> $kwargs_path

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
visual_main_root=$script_root/sequence_annotation/visual
coordinate_consist_bed_path=$saved_root/coordinate_consist.bed

echo "Step 1: Calculate distance between sites"
if [ ! -e "$site_diff_root/site_diff.tsv" ]; then
    python3 $preprocess_main_root/calculate_distance_between_sites.py -b $transcript_bed_path -t $tss_path \
    -c $cleavage_site_path -s $site_diff_root
fi

echo "Step 2: Preprocess raw data"
python3 $preprocess_main_root/get_external_UTR.py -b $transcript_bed_path -s $saved_root

echo "Step 3: Classify TSS and CS data to belonging site"
python3 $preprocess_main_root/classify_site.py -o $transcript_bed_path  -g $tss_path -c $cleavage_site_path \
-s $saved_root -u $upstream_dist -d $downstream_dist \
-f $saved_root/external_five_UTR.bed -t $saved_root/external_three_UTR.bed

echo "Step 4: Get maximize signal site data located on external UTR"
command="$preprocess_main_root/consist_site.py --external_five_UTR_tss_path \
$saved_root/external_five_UTR_tss.gff3 \
--external_three_UTR_cs_path $saved_root/external_three_UTR_cleavage_site.gff3 \
--long_dist_tss_path $saved_root/long_dist_tss.gff3 \
--long_dist_cs_path $saved_root/long_dist_cleavage_site.gff3 -s $saved_root \
--transcript_tss_path $saved_root/transcript_tss.gff3 \
--transcript_cs_path $saved_root/transcript_cleavage_site.gff3"

if $remove_transcript_external_UTR_conflict; then
    command="${command} --remove_transcript_external_UTR_conflict"
fi

python3 $command

echo "Step 5: Create coordiante data based on origin data and site data"
python3 $preprocess_main_root/create_coordinate_data.py -g $saved_root/safe_tss.gff3 -c $saved_root/safe_cs.gff3 \
-t $id_convert_table_path --single_start_end -o $saved_root/coordinate.gff3 --stats_path $saved_root/coordinate_data_count.json

python3 $preprocess_main_root/create_coordinate_bed.py -i $transcript_bed_path \
-c $saved_root/coordinate.gff3 -t $id_convert_table_path -o $coordinate_consist_bed_path

python3 $preprocess_main_root/coordinate_compare.py -r $transcript_bed_path -c $coordinate_consist_bed_path \
-o $saved_root/coordinate_compared.gff3

coding_transcript_id_path=$saved_root/coding_transcript_id.txt
safe_tss_transcript_id_path=$saved_root/safe_tss_transcript_id.txt
safe_cs_transcript_id_path=$saved_root/safe_cs_transcript_id.txt
venn_path=$saved_root/venn.png

python3 $preprocess_main_root/get_coding_id.py -i $transcript_bed_path -o $coding_transcript_id_path

python3 $preprocess_main_root/get_intersection_id.py -v $venn_path -i $coding_transcript_id_path,$safe_tss_transcript_id_path,$safe_cs_transcript_id_path -n "Coding,including TSS,including CS"

num_input=$(wc -l < $transcript_bed_path )
num_raw_tss=$(sed "1d"  $tss_path | wc -l )
num_raw_cs=$(sed "1d" $cleavage_site_path | wc -l)
num_external_five_UTR_tss=$(sed "1d" $saved_root/external_five_UTR_tss.gff3 | wc -l)
num_external_three_UTR_cleavage_site=$(sed "1d" $saved_root/external_three_UTR_cleavage_site.gff3 | wc -l)
num_safe_tss=$(sed "1d" $saved_root/safe_tss.gff3 | wc -l)
num_safe_cs=$(sed "1d" $saved_root/safe_cs.gff3 | wc -l)
num_consist=$(wc -l < $coordinate_consist_bed_path )
   
echo "Input mRNA count: $num_input" > $saved_root/preprocess.stats
echo "Evidence TSS count: $num_raw_tss" >> $saved_root/preprocess.stats
echo "Evidence CS count: $num_raw_cs" >> $saved_root/preprocess.stats
echo "Evidence TSS located at external 5' UTR: $num_external_five_UTR_tss" >> $saved_root/preprocess.stats
echo "Evidence CS located at external 3' UTR count: $num_external_three_UTR_cleavage_site" >> $saved_root/preprocess.stats
echo "Evidence TSS located at external 5' UTR and is most significant signal: $num_safe_tss" >> $saved_root/preprocess.stats
echo "Evidence CS located at external 3' UTR and is most significant signal: $num_safe_cs" >> $saved_root/preprocess.stats
echo "The number of mRNAs with both TSSs and CSs site supported and are passed by filter: $num_consist" >> $saved_root/preprocess.stats
