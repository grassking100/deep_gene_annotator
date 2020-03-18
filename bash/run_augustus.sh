#!/bin/bash
## function print usage
usage(){
 echo "Usage: Pipeline run Augustus"
 echo "  Arguments:"
 echo "    -r  <string>  Root of augustus"
 echo "    -s  <string>  Root of saved result"
 echo "    -t  <string>  Training fasta"
 echo "    -g  <string>  Training answer in GFF format"
 echo "    -x  <string>  Testing fasta"
 echo "    -d  <string>  Testing answer in GFF format"
 echo "    -n  <string>  Species name"
 echo "    -f  <int>     Flanking distance around gene of each direction"
 echo "    -m  <string>  The genemodel to be used"
 echo "    -a  <bool>    Do output prediction by alternatives_from_sampling [default: false]"
 echo "    -h            Print help message and exit"
 echo "Example: bash run_augustus.sh -s saved_root -t train_fasta -g train_gff -x val_fasta -d val_gff -n arabidopsis_2020_02_28_${i} -f 398 -a -m partial -r /root/augustus"
 echo ""
}

while getopts r:s:t:g:x:d:n:f:m:ah option
 do
  case "${option}"
  in
   r )augustus_root=$OPTARG;;
   s )saved_root=$OPTARG;;
   t )train_fasta=$OPTARG;;
   g )train_gff=$OPTARG;;
   x )test_fasta=$OPTARG;;
   d )test_gff=$OPTARG;;
   n )species=$OPTARG;;
   f )flanking=$OPTARG;;
   m )genemodel=$OPTARG;;
   a )alternatives_from_sampling=on;;
   h )usage; exit 1;;
   : )echo "Option $OPTARG requires an argument"
      usage; exit 1
      ;;
   \?)echo "Invalid option: $OPTARG"
      usage; exit 1
      ;;
 esac
done

if [ ! "$augustus_root" ]; then
    echo "Missing option -r"
    usage
    exit 1
fi

if [ ! "$saved_root" ]; then
    echo "Missing option -s"
    usage
    exit 1
fi

if [ ! "$train_fasta" ]; then
    echo "Missing option -t"
    usage
    exit 1
fi

if [ ! "$train_gff" ]; then
    echo "Missing option -g"
    usage
    exit 1
fi

if [ ! "$test_fasta" ]; then
    echo "Missing option -x"
    usage
    exit 1
fi

if [ ! "$test_gff" ]; then
    echo "Missing option -d"
    usage
    exit 1
fi

if [ ! "$species" ]; then
    echo "Missing option -n"
    usage
    exit 1
fi
if [ ! "$flanking" ]; then
    echo "Missing option -f"
    usage
    exit 1
fi

if [ ! "$genemodel" ]; then
    echo "Missing option -m"
    usage
    exit 1
fi

if [ ! "$alternatives_from_sampling" ]; then
    alternatives_from_sampling=off
fi

data_root=$saved_root/$species
train_dir=$data_root/train
test_dir=$data_root/test
mkdir -p $saved_root
mkdir -p $data_root
mkdir -p $train_dir
mkdir -p $test_dir

cd $data_root

AUGUSTUS_SCRIPTS_ROOT=$augustus_root/scripts
export AUGUSTUS_CONFIG_PATH=$augustus_root/config
export augustus_bin_path=$augustus_root/bin
species_path=$AUGUSTUS_CONFIG_PATH/species/$species
train_gb=$data_root/train.gb
test_gb=$data_root/test.gb

echo "Create gb for training"
gff2gbSmallDNA.pl $train_gff $train_fasta $flanking $train_gb
echo "Create gb for testing"
gff2gbSmallDNA.pl $test_gff $test_fasta $flanking $test_gb

echo "Create paremeters for species"
rm -rf $species_path
$AUGUSTUS_SCRIPTS_ROOT/new_species.pl --species=$species
echo "Training"
etraining --species=$species $train_gb --UTR=on > train.out
echo "Get training metric"
augustus --species=$species $train_gb --UTR=on --alternatives-from-sampling=$alternatives_from_sampling --genemodel=$genemodel> train.predict.out
echo "Retrain hyperparamter"
optimize_augustus.pl  --cpus=8 --species=$species --UTR=on --metapars=$AUGUSTUS_CONFIG_PATH/species/$species/${species}_metapars.utr.cfg train.gb> optimize.out
echo "Train after hyperparameter optimizer"
etraining --species=$species $train_gb --UTR=on > train.final.out
echo "Predict"
augustus --species=$species $train_fasta --UTR=on --alternatives-from-sampling=$alternatives_from_sampling --genemodel=$genemodel --gff=on > $train_dir/train.final.predict.gff
augustus --species=$species $train_gb --UTR=on --alternatives-from-sampling=$alternatives_from_sampling --genemodel=$genemodel > $train_dir/train.final.predict.out
augustus --species=$species $test_fasta --UTR=on --alternatives-from-sampling=$alternatives_from_sampling --genemodel=$genemodel --gff=on > $test_dir/test.final.predict.gff
augustus --species=$species $test_gb --UTR=on --alternatives-from-sampling=$alternatives_from_sampling --genemodel=$genemodel > $test_dir/test.final.predict.out

cp -r -t $data_root $species_path
