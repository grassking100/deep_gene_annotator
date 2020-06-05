import sys,os
import torch
import numpy as np
import deepdish as dd
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.genome_handler.seq_container import AnnSeqContainer
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.process.data_generator import aug_seq


def aug_data(fasta,annotation):
    raw_data = {'data':{'inputs':fasta,'answers':annotation}}
    data = AnnSeqProcessor(annotation.ANN_TYPES).process(raw_data)['data']

    indice = list(range(len(data['inputs'])))
    np.random.seed(0)
    np.random.shuffle(indice)
    indice_list = []
    step = 16
    for start in range(0,len(data['inputs']),step):
        indice_list.append(indice[start:start+step])

    new_seqs = {}
    new_anns = AnnSeqContainer()
    new_anns.ANN_TYPES = annotation.ANN_TYPES
    for indice_ in indice_list:
        print(annotation.ANN_TYPES)
        print(indice_)
        ids = [data['ids'][index] for index in indice_]
        inputs = [torch.LongTensor(data['inputs'][index]) for index in indice_]
        answers = [torch.LongTensor(data['answers'][index]) for index in indice_]
        lengths = [data['lengths'][index] for index in indice_]
        seqs = [data['seqs'][index] for index in indice_]
        has_gene_statuses = [data['has_gene_statuses'][index] for index in indice_]
        data_ = ids,inputs,answers,lengths,seqs,has_gene_statuses
        print(answers[0].shape)
        aug = aug_seq(data_,discard_ratio_min=0,discard_ratio_max=0.5,
                augment_up_max=100,augment_down_max=10,
                concat=True,shuffle=False)
        ids,inputs,answers,lengths,seqs,has_gene_statuses = aug
        for id_,seq,answer in zip(ids,seqs,answers):
            print(answer.shape)
            answer = answer.numpy()
            ann_seq = AnnSequence()
            ann_seq.ANN_TYPES = annotation.ANN_TYPES
            ann_seq.id = id_
            ann_seq.chromosome_id = id_
            ann_seq.length = len(seq)
            ann_seq.strand = 'plus'
            ann_seq.source = 'augmentation'
            ann_seq.init_space()
            for index,type_ in enumerate(ann_seq.ANN_TYPES):
                ann_seq.set_ann(type_,answer.T[index])
            new_seqs[id_] = seq
            new_anns[id_] = ann_seq
    return new_seqs,new_anns#.to_dict()


def main(seq_and_annotation_path,output_path):
    #fasta = read_fasta(fasta_path)
    fasta,annotation = dd.io.load(seq_and_annotation_path)
    annotation = AnnSeqContainer().from_dict(annotation)
    returned = aug_data(fasta,annotation)
    returned = returned[0],returned[1].to_dict()
    dd.io.save(output_path,returned)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i","--seq_and_annotation_path",required=True)
    parser.add_argument("-o","--output_path",required=True)

    args = parser.parse_args()
    kwargs = dict(vars(args))
    
    main(**kwargs)
