#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline creating annotation data"
 echo "  Arguments:"
 echo "    -a  <string>  Answer GFF path"
 echo "    -p  <string>  Output GFF path"
 echo "    -o  <string>  Directory of output folder"
 echo "    -h            Print help message and exit"
 echo "Example: bash gffcompare.sh -a answer.gff --p predict.gff -o data/2019_07_12"
 echo ""
}

while getopts a:p:o:h option
 do
  case "${option}"
  in
   a )answer_gff_path=$OPTARG;;
   p )predict_gff_path=$OPTARG;;
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

if [ ! "$answer_gff_path" ]; then
    echo "Missing option -a"
    usage
    exit 1
fi

if [ ! "$predict_gff_path" ]; then
    echo "Missing option -p"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

preprocess_root=/home/sequence_annotation/sequence_annotation/preprocess
predict_bed_path=$saved_root/predict.bed
predict_canonical_bed_path=$saved_root/predict_canonical.bed
predict_canonical_gff_path=$saved_root/predict_canonical.gff
predict_gene_gff_path=$saved_root/predict_gene.gff
id_convert_table_path=$saved_root/id_convert_table.tsv

mkdir -p $saved_root
#python3 $preprocess_root/bed2gff.py -i $answer_bed_path -o $answer_gff_path
python3 $preprocess_root/gff2bed.py -i $predict_gff_path -o $predict_bed_path
python3 $preprocess_root/get_id_table.py -i $predict_gff_path -o $id_convert_table_path
python3 $preprocess_root/path_decode.py -i $predict_bed_path -o $predict_canonical_gff_path --select_site_by_election -t $id_convert_table_path
python3 $preprocess_root/gff2bed.py -i $predict_canonical_gff_path -o $predict_canonical_bed_path
python3 $preprocess_root/bed2gff.py -i $predict_canonical_bed_path -o $predict_gene_gff_path
cd $saved_root
gffcompare --strict-match --debug --no-merge -T -r $answer_gff_path $predict_gene_gff_path 
