import os
import torch
from ..utils.utils import create_folder, read_fasta, write_bed, write_fasta
from ..utils.translate import translate
from ..preprocess.gff2bed import gff2bed
from .convert_signal_to_gff import simple_output_to_vectors

def simple_output_to_gff(seq_data,outputs,inference,ann_vec_gff_converter):
    outputs = {'outputs': outputs.detach(),'chrom_ids': seq_data.ids,'lengths': seq_data.lengths}
    output = simple_output_to_vectors(outputs)
    onehot_vecs = inference(output['outputs'],output['masks'])
    onehot_vecs = onehot_vecs.cpu().numpy()
    gff = ann_vec_gff_converter.convert(output['chrom_ids'],output['lengths'], onehot_vecs)
    gff = gff.sort_values(by=['chr','start','end','strand'])
    return gff

def gff_to_peptide(gff,genome_path,output_root):
    create_folder(output_root)
    bed_path = os.path.join(output_root,'data.bed')
    cDNA_path = os.path.join(output_root,'cDNA.fasta')
    os.path.join(output_root,'peptide.fasta')
    os.path.join(output_root,'orf_indice.json')
    bed = gff2bed(gff)
    write_bed(bed,bed_path)
    os.system("rm {}.fai".format(genome_path))
    os.system("bedtools getfasta -name -s -split -fi {} -bed {} -fo {}".format(genome_path,bed_path,cDNA_path))
    cDNA_fasta = read_fasta(cDNA_path)
    orf_indice_dict,orfs,peps = translate(cDNA_fasta)
    return orf_indice_dict,bed,peps

def get_output_score(ann_vec_gff_converter,inference,
                     outputs,seq_data,output_root):
    N,C,L = outputs.shape
    genome_path = os.path.join(output_root,'genome.fasta')
    fasta = dict(zip(seq_data.ids,seq_data.seqs))
    write_fasta(fasta,genome_path)
    gff = simple_output_to_gff(seq_data,outputs,
                               inference,ann_vec_gff_converter)
    orf_indice_dict,bed,peps = gff_to_peptide(gff,genome_path,output_root)
    bed = bed.groupby('id')
    scores = [torch.ones(1,L)]*N
    #for id_,score in zip(seq_data.ids,scores):
    #    for pep_id,pep in peps.items():
    #        if id_ in pep_id:
    #            orf_start,orf_end = orf_indice_dict[pep_id]
    #            item = bed.get_group(pep_id)
    #            item_start = list(item['start'])[0] - 1
    #            start = item_start + orf_start
    #            end = item_start + len(pep) - (orf_end+1)
    #            score[0,start:end+1] = 0
    scores = torch.cat(scores,0).cuda()
    #print(scores.mean())
    return scores

class ScoreCalculator:
    def __init__(self,peptide_path,ann_vec_gff_converter,inference):
        self.peptide_path = peptide_path
        self.ann_vec_gff_converter = ann_vec_gff_converter
        self.inference = inference
        
    def get_score(self,outputs,seq_data,output_root):
        score = get_output_score(self.ann_vec_gff_converter,self.inference,
                                 outputs,seq_data,output_root)
        #return score

