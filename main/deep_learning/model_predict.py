import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, read_fasta,read_json
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES
from sequence_annotation.preprocess.gff2bed import main as gff2bed_main
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine,get_best_model_and_origin_executor,get_batch_size
from sequence_annotation.postprocess.gff_reviser import main as revised_main
from main.utils import backend_deterministic

def predict(saved_root,model,executor,fasta,batch_size=None,**kwargs):
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.set_root(saved_root,with_train=False,with_val=False,
                    with_test=False,create_tensorboard=False,
                    with_predict=True)
    worker = engine.predict(model,executor,fasta,batch_size=batch_size,**kwargs)
    return worker

def main(trained_root,revised_root,output_root,fasta_path,region_table_path,
         deterministic=False,**kwargs):
    fasta = read_fasta(fasta_path)
    create_folder(trained_root)
    backend_deterministic(deterministic)
    batch_size = get_batch_size(trained_root)
    best_model,origin_executor = get_best_model_and_origin_executor(trained_root)
    predict(output_root,best_model,origin_executor,fasta,batch_size=batch_size,
            region_table_path=region_table_path,**kwargs)
    
    revised_config_path = os.path.join(revised_root,'best_gff_reviser_config.json')
    revised_config = read_json(revised_config_path)
    del revised_config['class']
    del revised_config['gene_info_extractor']
    del revised_config['ann_types']
    revised_root = os.path.join(output_root,'revised')
    raw_plus_gff = os.path.join(output_root,'predict','predict_raw_plus.gff3')
    plus_revised_gff_path = os.path.join(revised_root,'plus_revised.gff3')
    plus_revised_bed_path = os.path.join(revised_root,'plus_revised.bed')
    predicted_transcript_fasta_path = os.path.join(revised_root,'predicted_transcript.fasta')
    revised_main(revised_root, raw_plus_gff, region_table_path, fasta_path,**revised_config)
    gff2bed_main(plus_revised_gff_path,plus_revised_bed_path,simple_mode=False)
    os.system("bedtools getfasta -fi {} -bed {} -fo {} -name -split".format(fasta_path,plus_revised_bed_path,
                                                                    predicted_transcript_fasta_path))

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-d","--trained_root",help="Root of saved deep learning model",required=True)
    parser.add_argument("-f","--fasta_path",help="Path of fasta",required=True)
    parser.add_argument("-o","--output_root",help="The path to save testing result",required=True)
    parser.add_argument("-r","--revised_root",help="The root of revised result",required=True)
    parser.add_argument("-t","--region_table_path",help="Path of region table",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--deterministic",action="store_true")
    
    args = parser.parse_args()
    setting = vars(args)
    kwargs = dict(setting)
    del kwargs['gpu_id']
        
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
