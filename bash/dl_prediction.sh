#!/bin/bash
# function print usage
usage(){
 echo "Usage: The pipeline predict and revise transcript potential region by deep learning model"
 echo "  Arguments:"
 echo "    -i  <string>  Root of trained deep learning"
 echo "    -v  <int>     Root of Revised result"
 echo "    -r  <int>     Root of transcript potential region"
 echo "    -o  <string>  Directory for output result"
 echo "    -t  <string>  Directory of TransDecoder"
 echo "  Options:"
 echo "    -h            Print help message and exit"
 echo "Example: bash dl_prediction.sh -i trained_root -v revised -r region -o result -t transdecoder"
 echo ""
}

while getopts i:v:r:o:t:h option
 do
  case "${option}"
  in
   i )trained_dl_root=$OPTARG;;
   v )revised_root=$OPTARG;;
   r )region_root=$OPTARG;;
   o )saved_root=$OPTARG;;
   t )transdecoder_root=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$trained_dl_root" ]; then
    echo "Missing option -i"
    usage
    exit 1
fi

if [ ! "$revised_root" ]; then
    echo "Missing option -v"
    usage
    exit 1
fi

if [ ! "$region_root" ]; then
    echo "Missing option -r"
    usage
    exit 1
fi


if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$transdecoder_root" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

bash_root=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
script_root=$bash_root/..

mkdir -p $saved_root

kwargs_path=$saved_root/process_kwargs.csv
echo "name,value" > $kwargs_path
echo "trained_dl_root,$trained_dl_root" >> $kwargs_path
echo "revised_root,$revised_root" >> $kwargs_path
echo "region_root,$region_root" >> $kwargs_path
echo "saved_root,$saved_root" >> $kwargs_path
echo "transdecoder_root,$transdecoder_root" >> $kwargs_path

python3 $script_root/main/deep_learning/model_predict.py -d $trained_root -f $region_root/region.fasta -o $saved_root -t $region_root/region_id_conversion.tsv -r $revised_root -g 1 -e $region_root/region_double_strand.fasta

transdecoder_result_root=$saved_root/transdecoder_from_ds
mkdir -p $transdecoder_result_root
cd $transdecoder_result_root
perl $transdecoder_root/TransDecoder.LongOrfs -S -t $saved_root/revised/predicted_transcript_cDNA_from_ds_bed.fasta > $transdecoder_result_root/long_orf_record.log
perl $transdecoder_root/TransDecoder.Predict --single_best_only -t $saved_root/revised/predicted_transcript_cDNA_from_ds_bed.fasta > $transdecoder_result_root/predict_record.log
