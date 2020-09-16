import os
import sys
import torch
from argparse import ArgumentParser
from multiprocessing import cpu_count
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, read_fasta
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES
from sequence_annotation.preprocess.gff2bed import main as gff2bed_main
from sequence_annotation.process.director import SeqAnnEngine,get_best_model_and_origin_executor,get_batch_size
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.postprocess.gff_reviser import main as revised_main
from main.utils import backend_deterministic

def predict(saved_root,model,executor,seqs,region_table_path,
            batch_size=None,**kwargs):
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.batch_size = batch_size
    engine.set_root(saved_root,with_train=False,with_val=False,
                    with_test=False,create_tensorboard=False,
                    with_predict=True)
    #Set callbacks
    singal_handler = engine.get_signal_handler(saved_root,inference=executor.inference,
                                               region_table_path=region_table_path)
    callbacks = Callbacks()
    callbacks.add(singal_handler)
    #Set generator
    generator = engine.create_basic_data_gen()
    raw_data = {'prediction': {'inputs': seqs}}
    data = engine.process_data(raw_data)
    data_loader = generator(data['prediction'])
    worker = engine.predict(model,executor,data_loader,callbacks=callbacks,**kwargs)
    return worker

def main(trained_root,revised_root,output_root,fasta_path,fasta_double_strand_path,
         region_table_path,deterministic=False,batch_size=None,**kwargs):
    predict_root=os.path.join(output_root,'predict')
    predict_ps_gff_path = os.path.join(predict_root,'predict_plus_strand.gff3')
    if not os.path.exists(predict_ps_gff_path):
        fasta = read_fasta(fasta_path)
        create_folder(predict_root)
        backend_deterministic(deterministic)
        batch_size = batch_size or get_batch_size(trained_root)
        best_model,origin_executor = get_best_model_and_origin_executor(trained_root)
        predict(predict_root,best_model,origin_executor,fasta,region_table_path,
                batch_size=batch_size,**kwargs)
    
    revised_config_path = os.path.join(revised_root,'best_gff_reviser_config.json')
    revised_root = os.path.join(output_root,'revised')
    revised_ps_gff_path = os.path.join(revised_root,'revised_plus_strand.gff3')
    revised_ps_bed_path = os.path.join(revised_root,'revised_plus_strand.bed')
    if not os.path.exists(revised_ps_bed_path):
        revised_main(revised_root, predict_ps_gff_path, region_table_path, fasta_path,
                     revised_config_path,multiprocess=cpu_count())
        gff2bed_main(revised_ps_gff_path,revised_ps_bed_path,simple_mode=False)
    cDNA_fasta_path = os.path.join(revised_root,'predicted_transcript_cDNA.fasta')
    os.system("bedtools getfasta -fi {} -bed {} "
              "-fo {} -name -split".format(fasta_path,
                                           revised_ps_bed_path,
                                           cDNA_fasta_path))
    revised_ds_bed_path = os.path.join(revised_root, "revised_double_strand.bed")
    cDNA_fasta_from_ds_path = os.path.join(revised_root,'predicted_transcript_cDNA_from_ds_bed.fasta')
    os.system("bedtools getfasta -s -fi {} -bed {} "
              "-fo {} -name -split".format(fasta_double_strand_path,
                                           revised_ds_bed_path,
                                           cDNA_fasta_from_ds_path))

if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument("-d","--trained_root",help="Root of saved deep learning model",required=True)
    parser.add_argument("-f","--fasta_path",help="Path of single-strand fasta",required=True)
    parser.add_argument("-e","--fasta_double_strand_path",help="Path of double-strand fasta",required=True)
    parser.add_argument("-o","--output_root",help="The path to save testing result",required=True)
    parser.add_argument("-r","--revised_root",help="The root of revised result",required=True)
    parser.add_argument("-t","--region_table_path",help="Path of region table",required=True)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("-b","--batch_size",type=int,default=None)
    parser.add_argument("--deterministic",action="store_true")
    
    args = parser.parse_args()
    setting = vars(args)
    kwargs = dict(setting)
    del kwargs['gpu_id']
        
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
