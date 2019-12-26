import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..utils.utils import get_gff_with_attribute
from ..genome_handler.sequence import AnnSequence 
from ..genome_handler.ann_seq_processor import get_background,seq2vecs
from ..preprocess.utils import RNA_TYPES,GENE_TYPES,EXON_TYPES
from ..process.metric import calculate_metric,contagion_matrix,categorical_metric

GENE_ANN_TYPES = ['exon','intron','other']

def _normalize_matrix(matrix):
    normed_matrix = []
    for row in matrix:
        normed_matrix.append([item/sum(row) for item in row])
    return normed_matrix

def _create_ann_seq(chrom,length,strand):
    ann_seq = AnnSequence(['gene']+GENE_ANN_TYPES,length)
    ann_seq.chromosome_id = chrom
    ann_seq.strand = strand
    ann_seq.id = '{}_{}'.format(chrom,strand)
    return ann_seq

def _get_gff_with_type_coord_id(gff):
    part_gff = gff[['feature','chr','strand','start','end']]
    coord_id = part_gff.apply(lambda x: '_'.join([str(item) for item in x]), axis=1)
    gff = gff.assign(coord_id=coord_id)
    return gff

def _get_subgroup(gff,group_types):
    group = {}
    gene_ids = gff[gff['feature'].isin(GENE_TYPES)]['id']
    for gene_id in gene_ids:
        transcript_ids = gff[gff['parent']==gene_id]['id']
        if len(transcript_ids) != 1:
            raise Exception()
        transcript_id = list(transcript_ids)[0]
        group_ = gff[(gff['feature'].isin(group_types)) & (gff['parent']==transcript_id)].sort_values('start')
        group[gene_id] = group_
    return group

def _set_gene_intron_status(gff):
    group = {}
    gene_ids = gff[gff['feature'].isin(GENE_TYPES)]['id']
    for gene_id in gene_ids:
        transcript_ids = gff[gff['parent']==gene_id]['id']
        if len(transcript_ids) != 1:
            raise Exception()
        transcript_id = list(transcript_ids)[0]
        group_ = gff[(gff['feature']=='intron') & (gff['parent']==transcript_id)].sort_values('start')
        gff.loc[gff['id']==gene_id,'has_intron'] = len(group_)>0
    return group

def _compare_block(predict,answer,types):
    predict_coord_id = set(predict[predict['feature'].isin(types)]['coord_id'])
    answer_coord_id = set(answer[answer['feature'].isin(types)]['coord_id'])
    TP_ids = answer_coord_id.intersection(predict_coord_id)
    predict.loc[predict['coord_id'].isin(TP_ids),'match'] = True
    answer.loc[answer['coord_id'].isin(TP_ids),'match'] = True
    #print(predict[predict['id'].isin(TP_ids)])

def _match_gene(gff):
    exon_group = _get_subgroup(gff,EXON_TYPES)
    for gene_id,group in exon_group.items():
        if group['match'].all() and len(group)>0:
            gff.loc[gff['id']==gene_id,'match'] = True
            
def _match_intron_chain(gff):
    intron_group = _get_subgroup(gff,['intron'])
    for gene_id,group in intron_group.items():
        if group['match'].all() and len(group)>0:
            gff.loc[gff['id']==gene_id,'intron_chain_match'] = True
    
def _block_performance_per_type(predict,answer,type_name,types,match_key=None):
    match_key = match_key or 'match'
    predict_data = predict[predict['feature'].isin(types)]
    answer_data = answer[answer['feature'].isin(types)]
    TP_data = predict_data[predict_data[match_key]]
    FP_data = predict_data[~predict_data[match_key]]
    FN_data = answer_data[~answer_data['match']]
    FP_data = FP_data.assign(error_status= 'wrong {}'.format(type_name))
    FN_data = FN_data.assign(error_status= 'missing {}'.format(type_name))
    error_df = pd.concat([FP_data,FN_data],sort=True)
    error_df['attribute'] = error_df[['attribute',
                                      'error_status']].apply(lambda x: "{};Error={}".format(*x),
                                                             axis=1)
    TP = len(TP_data)
    FP = len(FP_data)
    FN = len(FN_data)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    F1=0
    if recall+precision > 0:
        F1 = (2*recall*precision)/(recall+precision)
    status = {}
    status['{}_TP'.format(type_name)] = TP
    status['{}_FP'.format(type_name)] = FP
    status['{}_FN'.format(type_name)] = FN
    status['{}_recall'.format(type_name)] = recall
    status['{}_precision'.format(type_name)] = precision
    status['{}_F1'.format(type_name)] = F1
    return status,error_df
   
def _compare_gene(predict,answer):
    _compare_block(predict,answer,EXON_TYPES)
    _compare_block(predict,answer,['intron'])
    _match_gene(predict)
    _match_gene(answer)
    _match_intron_chain(predict)
    _match_intron_chain(answer)

def _block_performance(predict,answer):
    predict = get_gff_with_attribute(predict)
    answer = get_gff_with_attribute(answer)
    predict = _get_gff_with_type_coord_id(predict)
    answer = _get_gff_with_type_coord_id(answer)
    answer['match']=False
    predict['match']=False
    answer['has_intron']=False
    predict['has_intron']=False
    answer['intron_chain_match']=False
    predict['intron_chain_match']=False
    _set_gene_intron_status(answer)
    _set_gene_intron_status(predict)
    _compare_gene(predict,answer)
    status = {}
    error_df = []
    intron_status,intron_error_df = _block_performance_per_type(predict,answer,'intron',['intron'])
    exon_status,exon_error_df = _block_performance_per_type(predict,answer,'exon',EXON_TYPES)
    gene_status,gene_error_df = _block_performance_per_type(predict,answer,'gene',GENE_TYPES)
    intorn_chain_performance = _block_performance_per_type(predict[predict['has_intron']],
                                                           answer[answer['has_intron']],
                                                           'intron_chain',GENE_TYPES,
                                                           match_key='intron_chain_match')
    intron_chain_status,intron_chain_error_df = intorn_chain_performance
    status.update(intron_status)
    status.update(exon_status)
    status.update(gene_status)
    status.update(intron_chain_status)
    error_df.append(intron_error_df)
    error_df.append(exon_error_df)
    error_df.append(gene_error_df)
    error_df.append(intron_chain_error_df)
    error_df = pd.concat(error_df)
    return status,error_df

def _gene_gff2vec(gff,chrom_id,strand,length):
    subgff = gff[(gff['chr']==chrom_id) & (gff['strand']==strand)]
    strand = 'plus' if strand == '+' else 'minus'
    ann_seq = _create_ann_seq(chrom_id,length,strand)
    for block in subgff[subgff['feature']=='gene'].to_dict('record'):
        ann_seq.add_ann('gene',1,block['start']-1,block['end']-1)
    for block in subgff[subgff['feature']=='exon'].to_dict('record'):
        ann_seq.add_ann('exon',1,block['start']-1,block['end']-1)
    other = get_background(ann_seq,['gene'])
    ann_seq.set_ann('other',other)
    ann_seq.op_not_ann('intron','gene','exon')
    ann_seq = ann_seq.get_subseq(ann_types=GENE_ANN_TYPES)
    vec = np.array([seq2vecs(ann_seq)]).transpose(0,2,1)
    return vec

def gff_performance(predict,answer,chrom_lengths,round_value=None):
    """
    The method would compare data between prediction and answer, and return the performance result
    
    Parameters
    ----------
    predict : pandas.DataFrame
        GFF dataframe of prediction
    answer : pandas.DataFrame
        GFF dataframe of answer
    chrom_lengths : dict
        The dictionary of chromosomes' lengths
    round_value : int, optional
        rounded at the specified number of digits, if it is None then the result wouldn't be rounded (default is None)
    Returns
    -------
    base_performance : dict
        The dictionary about base performance
    contagion_matrix_ : array
        The array about contagion matrix
    block_performance : dict
        The dictionary about gene and exon performance
    error_data : pd.DataFrame
        The dataframe about error gene and exon
    """
    label_num=len(GENE_ANN_TYPES)
    metric = {}
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    metric['T'] = 0
    metric['F'] = 0
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    contagion_matrix_ = np.array([[0]*label_num]*label_num)
    chrom_ids = list(chrom_lengths.keys())
    for chrom_id in chrom_ids:
        length = chrom_lengths[chrom_id]
        for strand in ['+','-']:
            predict_vec = _gene_gff2vec(predict,chrom_id,strand,length)
            answer_vec = _gene_gff2vec(answer,chrom_id,strand,length)
            mask = np.ones((1,length))
            metric_ = categorical_metric(predict_vec,answer_vec,mask)
            contagion_matrix_ += np.array(contagion_matrix(predict_vec,answer_vec,mask))
            for type_ in ['TPs','FPs','TNs','FNs']:
                for index in range(label_num):
                    metric[type_][index] += metric_[type_][index]
            metric['T'] += metric_['T']
            metric['F'] += metric_['F']
    base_performance = calculate_metric(metric,label_names=GENE_ANN_TYPES,round_value=round_value)
    block_performance,error_status = _block_performance(predict,answer)
    return base_performance,contagion_matrix_,block_performance,error_status

def draw_contagion_matrix(contagion_matrix,round_number=None):
    """
    The method would draw contagion matrix which are normalized by row
    
    Parameters
    ----------
    contagion_matrix : array
        A 2D array of contagion matrix
    round_value : int, optional
        rounded at the specified number of digits (default is 3)
    """
    round_number = round_number or 3
    normed = _normalize_matrix(contagion_matrix)
    plt.matshow(normed)
    format_ = "{:."+str(round_number)+"f}"
    for r_index,row in enumerate(normed):
        for c_index,cell in enumerate(row):
            plt.text(c_index,r_index, format_.format(cell), ha='center', va='center')
    plt.show()
