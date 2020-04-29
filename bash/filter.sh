#!/bin/bash
usage(){
 echo "Usage: Pipeline filtering annotation data"
 echo "  Arguments:"
 echo "    -i  <string>  Path of bed"
 echo "    -t  <string>  Gene and transcript id conversion table path"
 echo "    -o  <string>  Directory for output"
 echo "  Options:"
 echo "    -a  <bool>    Remove gene with altenative donor site and acceptor site                             [default: false]"
 echo "    -x  <bool>    Remove gene with non-coding transcript                                               [default: false]"
 echo "    -l  <string>  Path of gene id list to be preserved"
 echo "    -n  <string>  Name of preserved gene id list                                                   [default: Preserved]"
 echo "    -z  <str>     Mode to select for comparing score, valid options are 'bigger_or_equal', "
 echo "                  'smaller_or_equal'                                                          [default:bigger_or_equal]"
 echo "    -f  <float>   BED item to preserved when comparing score and threshold, defualt would ignore score"
 echo "    -y  <float>   If it is true, then the gene with transcript which has failed to passed the score "
 echo "                  filter would be removed. Otherwise, only the transcript which has failed to passed "
 echo "                  the score filter would be removed                                                    [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash filter.sh -i /home/io/example.bed -o filtered_result -t table.tsv"
 echo ""
}

while getopts i:t:o:f:l:n:z:axyh option
 do
  case "${option}"
  in
   i )transcript_bed_path=$OPTARG;;
   t )id_convert_table_path=$OPTARG;;
   o )saved_root=$OPTARG;;
   f )score_filter=$OPTARG;;
   l )preserved_gene_id_path=$OPTARG;;
   n )preserved_id_name=$OPTARG;;
   z )compared_mode=$OPTARG;;
   a )remove_alt_site=true;;
   x )remove_non_coding=true;;
   y )remove_fail_score_gene=true;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$transcript_bed_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$id_convert_table_path" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

if [ ! "$remove_alt_site" ]; then
    remove_alt_site=false
fi

if [ ! "$remove_non_coding" ]; then
    remove_non_coding=false
fi

if [ ! "$remove_fail_score_gene" ]; then
    remove_fail_score_gene=false
fi

if [ ! "$compared_mode" ]; then
    compared_mode=bigger_or_equal
fi

if [ ! "$preserved_id_name" ]; then
    preserved_id_name=Preserved
fi

rm -rf $saved_root
mkdir -p $saved_root

kwargs_path=$saved_root/filter_kwargs.csv
echo "name,value" > $kwargs_path
echo "transcript_bed_path,$transcript_bed_path" >> $kwargs_path
echo "id_convert_table_path,$id_convert_table_path" >> $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path
echo "score_filter,$score_filter" >> $kwargs_path
echo "compared_mode,$compared_mode" >> $kwargs_path
echo "remove_fail_score_gene,$remove_fail_score_gene" >> $kwargs_path
echo "remove_alt_site,$remove_alt_site" >> $kwargs_path
echo "remove_non_coding,$remove_non_coding" >> $kwargs_path
echo "preserved_gene_id_path,$preserved_gene_id_path" >> $kwargs_path
echo "preserved_id_name,$preserved_id_name" >> $kwargs_path

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..
preprocess_main_root=$script_root/sequence_annotation/preprocess
visual_main_root=$script_root/sequence_annotation/visual

transcript_id_path=$saved_root/transcript_id.txt
python3 $preprocess_main_root/get_id.py -i $transcript_bed_path -o $transcript_id_path

echo "Step 1: Remove gene or transcript which doesn't pass threshold filter"
score_filter_root=$saved_root/score_filter
passed_score_bed_path=$score_filter_root/passed_score_filter.bed
passed_score_id_path=$score_filter_root/passed_score_id.txt
mkdir -p $score_filter_root

if [ "$score_filter" ]; then
    command="$preprocess_main_root/bed_score_filter.py -i $transcript_bed_path -o $passed_score_bed_path --threshold $score_filter --mode $compared_mode -t $id_convert_table_path"
    if $remove_fail_score_gene ; then
        command="$command --remove_gene"
    fi
    echo "python3 $command"
    python3 $command
    python3 $preprocess_main_root/get_id.py -i $passed_score_bed_path -o $passed_score_id_path
else
    python3 $preprocess_main_root/get_id.py -i $transcript_bed_path -o $passed_score_id_path
fi

echo "Step 2: Remove overlapped gene on the same strand"
nonoverlap_filter_root=$saved_root/nonoverlap_filter
nonoverlap_bed_path=$nonoverlap_filter_root/nonoverlap.bed
nonoverlap_id_path=$nonoverlap_filter_root/nonoverlap_id.txt
mkdir -p $nonoverlap_filter_root
python3 $preprocess_main_root/nonoverlap_filter.py -i $transcript_bed_path -s $nonoverlap_filter_root --use_strand \
-t $id_convert_table_path 
python3 $preprocess_main_root/get_id.py -i $nonoverlap_bed_path -o $nonoverlap_id_path

echo "Step 3 (Optional): Remove transcript which its gene has alternative donor or acceptor site"
remove_alt_site_root=$saved_root/remove_alt_site
no_alt_site_bed_path=$remove_alt_site_root/no_alt_site.bed
no_alt_site_id_path=$remove_alt_site_root/no_alt_site_id.txt
mkdir -p $remove_alt_site_root
if $remove_alt_site ; then
    python3 $preprocess_main_root/remove_alt_site.py -i $transcript_bed_path \
    -o $no_alt_site_bed_path -t $id_convert_table_path
    python3 $preprocess_main_root/get_id.py -i $no_alt_site_bed_path -o $no_alt_site_id_path
else
    python3 $preprocess_main_root/get_id.py -i $transcript_bed_path -o $no_alt_site_id_path
fi

echo "Step 4 (Optional): Remove transcript which its gene has non-coding transcript"
remove_non_coding_root=$saved_root/remove_non_coding
all_coding_bed_path=$remove_non_coding_root/all_coding.bed
all_coding_id_path=$remove_non_coding_root/all_coding_id.txt
mkdir -p $remove_non_coding_root
if $remove_non_coding ; then
    python3 $preprocess_main_root/belonging_gene_coding_filter.py -i $transcript_bed_path -t $id_convert_table_path \
    -o $all_coding_bed_path
    python3 $preprocess_main_root/get_id.py -i $all_coding_bed_path -o $all_coding_id_path
else
    python3 $preprocess_main_root/get_id.py -i $transcript_bed_path -o $all_coding_id_path
fi

echo "Step 5: Get intersection of transcript id and $preserved_id_name id"
if [ "$preserved_gene_id_path" ]; then
    intersection_preserved_id_path=$saved_root/intersection_preserved_id.txt
    preserved_transcript_id_path=$saved_root/preserved_transcript_id.txt
    python3 $preprocess_main_root/get_transcript_id.py -i $preserved_gene_id_path -o $preserved_transcript_id_path -t $id_convert_table_path
    python3 $preprocess_main_root/get_intersection_id.py -o $intersection_preserved_id_path -i $transcript_id_path,$preserved_transcript_id_path -n "All,$preserved_id_name"
fi

echo "Step 6: Get data which its gene passed filters"
intersection_id_path=$saved_root/intersection_id.txt
filtered_bed_path=$saved_root/filtered.bed
ven_path=$saved_root/venn.png
if [ "$preserved_gene_id_path" ]; then
    python3 $preprocess_main_root/get_intersection_id.py -o $intersection_id_path -v $ven_path \
-i $passed_score_id_path,$nonoverlap_id_path,$no_alt_site_id_path,$all_coding_id_path,$intersection_preserved_id_path \
-n "Passed score,Nonoverlapped with other gene,No-alternative splicing site,All-Coding,$preserved_id_name"

else
    python3 $preprocess_main_root/get_intersection_id.py -o $intersection_id_path -v $ven_path -i $passed_score_id_path,$nonoverlap_id_path,$no_alt_site_id_path,$all_coding_id_path -n "Passed score,Nonoverlapped with other gene,No-alternative splicing site,All-Coding"
fi

python3 $preprocess_main_root/get_subbed.py -i $transcript_bed_path -d $intersection_id_path -o $filtered_bed_path
