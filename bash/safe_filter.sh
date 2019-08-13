#!/bin/bash
## function print usage
usage(){
 echo "Usage: Get regions in wanted regions which are not overlap with unwanted regions in specific distance"
 echo "  Arguments:"
 echo "    -i  <string>  Wanted region bed path"
 echo "    -x  <string>  Unwanted region bed path"
 echo "    -f  <string>  Fai file path"
 echo "    -u  <int>  Upstream distance"
 echo "    -d  <int>  Downstream distance"
 echo "    -o  <int>  Output path"
 echo "  Options:"
 echo "    -s  <bool>    Filter with same strand [default: false]"
 echo "    -h  Print help message and exit"
 echo "Example: bash safe_filter.sh -i want.bed -x unwant.bed -u 1000 -d 500 -f genime.fai -o result.bed"
 echo ""
}
while getopts i:x:f:u:d:o:sh option
 do
  case "${option}"
  in
   i )want_path=$OPTARG;;
   x )unwant_path=$OPTARG;;
   f )fai_path=$OPTARG;;
   u )upstream_dist=$OPTARG;;
   d )downstream_dist=$OPTARG;;
   o )output_path=$OPTARG;;
   s )use_strand=true;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$want_path" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi
if [ ! "$unwant_path" ]; then
    echo "Missing option -x"
    usage
    exit 1
fi
if [ ! "$fai_path" ]; then
    echo "Missing option -f"
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

if [ ! "$output_path" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$use_strand" ]; then
    use_strand=false
fi

script_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
expand_path=_region_up_${upstream_dist}_down_${downstream_dist}.bed
saved_path=_region_up_${upstream_dist}_down_${downstream_dist}_safe_zone.bed

echo Expand around $want_path
bedtools slop -s -i $want_path -g $fai_path -l $upstream_dist -r $downstream_dist > $expand_path
#Output safe zone
echo Output safe zone of $want_path
if $use_strand ; then
    bedtools intersect -s -a $expand_path -b $unwant_path -wa -v > $saved_path
else    
    bedtools intersect -a $expand_path -b $unwant_path -wa -v > $saved_path
fi    
echo Get id
bash $script_root/get_ids.sh -i $saved_path > $output_path

rm $saved_path
rm $expand_path
