#!/bin/bash
usage(){
 echo "Usage: The pipeline ro run busco"
 echo "  Arguments:"
 echo "    -f  <string>  Path of sequence in fasta format"
 echo "    -n  <string>  Name"
 echo "    -l  <string>  Lineage name"
 echo "    -o  <string>  Root to save result"
 echo "    -m  <string>  Mode"
 echo "  Options:"
 echo "    -c  <int>     Number of CPU                   [default:40]"
 echo "    -h            Print help message and exit"
 echo "Example: bash run_busco.sh -f /home/io/example.fasta -o canoncial_result -n test -l brassicales_odb10 -m protein"
 echo ""
}

while getopts f:l:o:n:m:c:h option
 do
  case "${option}"
  in
   f )fasta_path=$OPTARG;;
   l )lineage_name=$OPTARG;;
   o )output_root=$OPTARG;;
   n )name=$OPTARG;;
   c )ncpu=$OPTARG;;
   m )mode=$OPTARG;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$fasta_path" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi

if [ ! "$lineage_name" ]; then
    echo "Missing option -l"
    usage
    exit 1
fi


if [ ! "$output_root" ]; then
    echo "Missing option -o"
    usage
    exit 1
fi

if [ ! "$name" ]; then
    echo "Missing option -n"
    usage
    exit 1
fi

if [ ! "$mode" ]; then
    echo "Missing option -m"
    usage
    exit 1
fi

if [ ! "$ncpu" ]; then
    ncpu=40
fi

mkdir -p $output_root
result_root=$output_root/$name
mkdir -p $result_root

kwargs_path=$result_root/process_kwargs.csv
echo "name,value" > $kwargs_path
echo "fasta_path,$fasta_path" >> $kwargs_path
echo "lineage_name,$lineage_name" >> $kwargs_path
echo "output_root,$output_root" >> $kwargs_path
echo "name,$name" >> $kwargs_path
echo "ncpu,$ncpu" >> $kwargs_path

cp -$fasta_path $result_root/seq.fasta
cd $result_root
docker run -u $(id -u) -v $(pwd):/busco_wd ezlabgva/busco:v4.0.6_cv1 busco -m $mode -i seq.fasta -o $name -l $lineage_name -c $ncpu
