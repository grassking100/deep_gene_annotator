import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from  matplotlib import pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../..")))
from sequence_annotation.utils.utils import create_folder, write_gff, write_json, read_gff, read_region_table,BASIC_GENE_ANN_TYPES
from sequence_annotation.utils.site_analysis import site_abs_diff
from sequence_annotation.genome_handler.sequence import AnnSequence 
from sequence_annotation.genome_handler.ann_seq_processor import get_background,seq2vecs
from sequence_annotation.preprocess.utils import EXON_TYPES, GENE_TYPES
from sequence_annotation.preprocess.utils import get_gff_with_attribute,get_gff_with_feature_coord,get_gff_with_intron
from sequence_annotation.process.metric import calculate_metric,contagion_matrix,categorical_metric

def _normalize_matrix(matrix):
    normed_matrix = []
    for row in matrix:
        normed_matrix.append([item/sum(row) for item in row])
    return normed_matrix

def _create_ann_seq(chrom,length,strand):
    ann_seq = AnnSequence(['gene']+BASIC_GENE_ANN_TYPES,length)
    ann_seq.chromosome_id = chrom
    ann_seq.strand = strand
    ann_seq.id = '{}_{}'.format(chrom,strand)
    return ann_seq

def _get_group_dict(gff,group_types):
    group = {}
    gene_ids = gff[gff['feature'].isin(GENE_TYPES)]['id']
    for gene_id in gene_ids:
        transcript_ids = list(gff[gff['parent']==gene_id]['id'])
        if len(transcript_ids) != 1:
            raise Exception("{} has multiple transcripts {}".format(gene_id,transcript_ids))
        transcript_id = transcript_ids[0]
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
    
def _set_gene_internal_exon_status(gff):
    exon_group_dict = _get_group_dict(gff,EXON_TYPES)
    for gene_id,group in exon_group_dict.items():
        if len(group)>=3:
            ids = list(group.sort_values('start')['feature_coord'][1:-1])
            gff.loc[gff['feature_coord'].isin(ids),'is_internal_exon'] = True

def _compare_block(predict,answer,types):
    predict_feature_coord = set(predict[predict['feature'].isin(types)]['feature_coord'])
    answer_feature_coord = set(answer[answer['feature'].isin(types)]['feature_coord'])
    TP_ids = answer_feature_coord.intersection(predict_feature_coord)
    for id_ in TP_ids:
        predict_gene = list(predict[predict['feature_coord']==id_]['belong_gene'])[0]
        answer_gene = list(answer[answer['feature_coord']==id_]['belong_gene'])[0]
        predict.loc[predict['feature_coord']==id_,'match'] = True
        answer.loc[answer['feature_coord']==id_,'match'] = True
        predict.loc[predict['feature_coord']==id_,'partner_gene'] = answer_gene
        answer.loc[answer['feature_coord']==id_,'partner_gene'] = predict_gene

def _compare_internal_exon(predict,answer):
    predict_feature_coord = set(predict[predict['is_internal_exon']]['feature_coord'])
    answer_feature_coord = set(answer[answer['is_internal_exon']]['feature_coord'])
    TP_ids = answer_feature_coord.intersection(predict_feature_coord)
    predict.loc[predict['feature_coord'].isin(TP_ids),'internal_exon_match'] = True
    answer.loc[answer['feature_coord'].isin(TP_ids),'internal_exon_match'] = True
    
def _match_chain_block(subject,partner,types,set_match_key):
    subject_group_dict = _get_group_dict(subject,types)
    partner_group_dict = _get_group_dict(partner,types)
    for gene_id,subject_group in subject_group_dict.items():
        if len(subject_group)>0:
            partner_genes = list(subject_group['partner_gene'])
            partner_gene = partner_genes[0]
            if len(set(partner_genes)) == 1 and str(partner_gene)!='nan':
                partner_group = partner_group_dict[partner_gene]
                if set(partner_group['feature_coord'])==set(subject_group['feature_coord']):
                    subject.loc[subject['id']==gene_id,set_match_key] = True
    
def _compare_chain_block(predict,answer,types,set_match_key):
    _match_chain_block(predict,answer,types,set_match_key)
    _match_chain_block(answer,predict,types,set_match_key)

def _performance_calculate(TP,FP,FN,answer_num,predict_num,type_name,round_value=None):
    if (TP+FN)!=answer_num or (TP+FP)!=predict_num:
        print(type_name,TP,FN,FP,answer_num,predict_num)
        raise Exception("Wrong number in {}".format(type_name))
        
    recall = 0
    precision = 0
    F1=0
    if answer_num > 0:
        recall = TP/(TP+FN)
    if predict_num>0:
        precision = TP/(TP+FP)
    if recall is not None and precision is not None and recall+precision > 0:
        F1 = (2*recall*precision)/(recall+precision)
    status = {}
    status['predict_{}_number'.format(type_name)] = predict_num
    status['real_{}_number'.format(type_name)] = answer_num
    status['{}_TP'.format(type_name)] = TP
    status['{}_FP'.format(type_name)] = FP
    status['{}_FN'.format(type_name)] = FN
    status['{}_recall'.format(type_name)] = recall
    status['{}_precision'.format(type_name)] = precision
    status['{}_F1'.format(type_name)] = F1
    if round_value is not None:
        for key,value in status.items():
            if isinstance(value,float):
                status[key] = round(value,round_value)
    return status

def _block_performance(predict,answer,type_name,types,match_key=None,round_value=None):
    id_name = 'feature_coord'
    match_key = match_key or 'match'
    predict_data = predict[predict['feature'].isin(types)]
    answer_data = answer[answer['feature'].isin(types)]
    predict_ids = set(predict_data[id_name])
    answer_ids = set(answer_data[id_name])
    TP_predeict_ids = set(predict_data[predict_data[match_key]][id_name])
    TP_answer_ids = set(answer_data[answer_data[match_key]][id_name])
    if len(TP_predeict_ids) != len(TP_answer_ids):
        pc = set(predict_data[predict_data[match_key]]['chr'])
        ac = set(answer_data[answer_data[match_key]]['chr'])
        print(pc-ac,ac-pc)
        raise Exception("Inconsist number between {} and {} at {}".format(len(TP_predeict_ids),
                                                                          len(TP_answer_ids),
                                                                          type_name))
    FP_ids = predict_ids-TP_predeict_ids
    FN_ids = answer_ids-TP_answer_ids
    FP_data = predict_data[predict_data[id_name].isin(FP_ids)]
    FN_data = answer_data[answer_data[id_name].isin(FN_ids)]
    predict_num = len(predict_ids)
    answer_num = len(answer_ids)
    TP = len(TP_predeict_ids)
    FP = len(FP_ids)
    FN = len(FN_ids)
    FP_data = FP_data.assign(error_status= 'wrong {}'.format(type_name))
    FN_data = FN_data.assign(error_status= 'missing {}'.format(type_name))
    error_df = pd.concat([FP_data,FN_data],sort=True)
    if len(error_df) > 0:
        attributes = error_df[['attribute','error_status']].apply(lambda x: "{};Error={}".format(*x),
                                                                  axis=1)
        error_df['attribute'] = attributes
    status = _performance_calculate(TP,FP,FN,answer_num,predict_num,
                                    type_name,round_value=round_value)
    return status,error_df

def block_performance(predict,answer,round_value=None):
    predict = get_gff_with_attribute(predict)
    answer = get_gff_with_attribute(answer)
    predict = get_gff_with_feature_coord(predict)
    answer = get_gff_with_feature_coord(answer)
    predict = predict[~predict['feature_coord'].duplicated(keep='first')]    
    answer = answer[~answer['feature_coord'].duplicated(keep='first')]
    
    for gff in [answer,predict]:
        gff['belong_gene']=None
        gff['match']=False
        gff['has_intron']=False
        gff['intron_chain_match']=False
        gff['is_internal_exon']=False
        gff['internal_exon_match']=False
        _set_gene_internal_exon_status(gff)
        _set_gene_intron_status(gff)

        parent_dict = dict(zip(gff['id'],gff['parent']))
        parents = []
        is_exon_intron = gff['feature'].isin(EXON_TYPES+['intron'])
        for parent in gff[is_exon_intron]['parent']:
            parents.append(parent_dict[parent])
        gff.loc[is_exon_intron,'belong_gene'] = parents
        
    _compare_block(predict,answer,EXON_TYPES+['intron'])
    _compare_internal_exon(predict,answer)
    _compare_chain_block(predict,answer,EXON_TYPES,'match')
    _compare_chain_block(predict,answer,['intron'],'intron_chain_match')

    intron_status,intron_error_df = _block_performance(predict,answer,'intron',
                                                       ['intron'],round_value=round_value)

    exon_status,exon_error_df = _block_performance(predict,answer,'exon',EXON_TYPES,
                                                   round_value=round_value)

    gene_status,gene_error_df = _block_performance(predict,answer,'gene',GENE_TYPES,
                                                   round_value=round_value)

    intron_chain_performance = _block_performance(predict[predict['has_intron']],
                                                  answer[answer['has_intron']],
                                                  'intron_chain',GENE_TYPES,
                                                  match_key='intron_chain_match',
                                                  round_value=round_value)
    
    internal_exon_performance = _block_performance(predict[predict['is_internal_exon']],
                                                   answer[answer['is_internal_exon']],
                                                   'internal_exon',EXON_TYPES,
                                                   match_key='internal_exon_match',
                                                   round_value=round_value)
    
    intron_chain_status,intron_chain_error_df = intron_chain_performance
    internal_exon_status,internal_exon_error_df = internal_exon_performance
    status_list = [intron_status,exon_status,gene_status,intron_chain_status,internal_exon_status]
    
    status = {}
    for status_ in status_list:
        status.update(status_)
        
    error_df = [intron_error_df,exon_error_df,gene_error_df,
                intron_chain_error_df,internal_exon_error_df]
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
    ann_seq = ann_seq.get_subseq(ann_types=BASIC_GENE_ANN_TYPES)
    vec = np.array([seq2vecs(ann_seq)]).transpose(0,2,1)
    return vec

def gff_performance(predict,answer,region_table,round_value=None):
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
    site_p_a_status : dict
        A dictionary about median of absolute minimal difference of each predict site to answer site
    site_a_p_status : dict
        A dictionary about median of absolute minimal difference of each answer site to predict site
    """
    answer_list = answer.to_dict('record')
    region_table_list = region_table.to_dict('record')
    answer_regions = set()
    table_regions = set()
    for item in answer_list:
        answer_regions.add("{}_{}".format(item['chr'],item['strand']))
        
    for item in region_table_list:
        table_regions.add("{}_{}".format(item['new_id'],item['strand']))

    for answer_region in answer_regions:
        if answer_region not in table_regions:
            raise Exception(answer_region,table_regions)

    predict = get_gff_with_intron(predict)
    answer = get_gff_with_intron(answer)
    label_num=len(BASIC_GENE_ANN_TYPES)
    metric = {}
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    metric['T'] = 0
    metric['F'] = 0
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    contagion = np.array([[0]*label_num]*label_num)
    chrom_ids = list(chrom_lengths.keys())
    
    for item in region_table.to_dict('record'):
        chrom_id = item['new_id']
        length = item['length']
        strand = item['strand']
        predict_vec = _gene_gff2vec(predict,chrom_id,strand,length)
        answer_vec = _gene_gff2vec(answer,chrom_id,strand,length)
        mask = np.ones((1,length))
        metric_ = categorical_metric(predict_vec,answer_vec,mask)
        contagion += np.array(contagion_matrix(predict_vec,answer_vec,mask))
        for type_ in ['TPs','FPs','TNs','FNs']:
            for index in range(label_num):
                metric[type_][index] += metric_[type_][index]
        metric['T'] += metric_['T']
        metric['F'] += metric_['F']
    base_perform = calculate_metric(metric,label_names=BASIC_GENE_ANN_TYPES,round_value=round_value)
    block_perform,error_status = block_performance(predict,answer,round_value=round_value)
    site_p_a_status = site_abs_diff(answer,predict,round_value=round_value)
    site_a_p_status = site_abs_diff(answer,predict,answer_as_ref=False,round_value=round_value)
    return base_perform,contagion,block_perform,error_status,site_p_a_status,site_a_p_status

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

def site_diff_table(roots,names):
    site_diff = []
    error_paths = []
    for root,name in zip(roots,names):
        try:
            p_a_abs_diff = read_json(os.path.join(root,'p_a_abs_diff.json'))
            a_p_abs_diff = read_json(os.path.join(root,'a_p_abs_diff.json'))
        except:
            error_paths.append(root)
            continue
        for key,values in p_a_abs_diff.items():
            for target,value in values.items():
                site_diff_ = {}
                site_diff_['method'] = '{}(abs(predict-answer))'.format(key)
                site_diff_['target'] = target
                site_diff_['value'] = value
                site_diff_['name'] = name
                site_diff.append(site_diff_)

        for key,values in a_p_abs_diff.items():
            for target,value in values.items():
                site_diff_ = {}
                site_diff_['method'] = '{}(abs(answer-predict))'.format(key)
                site_diff_['target'] = target
                site_diff_['value'] = value
                site_diff_['name'] = name
                site_diff.append(site_diff_)
    site_diff = pd.DataFrame.from_dict(site_diff)[['name','target','method','value']]
    return site_diff,error_paths

def block_performance_table(roots,names):
    block_performance = []
    error_paths = []
    for root,name in zip(roots,names):
        try:
            data = read_json(os.path.join(root,'block_performance.json'))
        except:
            error_paths.append(root)
            continue
            
        for target,value in data.items():
            block_performance_ = {}
            block_performance_['target'] = target
            block_performance_['value'] = value
            block_performance_['name'] = name
            block_performance.append(block_performance_)

    block_performance = pd.DataFrame.from_dict(block_performance)
    columns = list(block_performance.columns)
    columns.remove('name')
    columns = ['name'] + columns
    return block_performance[columns],error_paths

def compare_and_save(predict,answer,region_table,saved_root,round_value=None):
    result = gff_performance(predict,answer,region_table,round_value)
    base_perform,contagion,block_perform,errors,site_p_a_diff,site_a_p_abs_diff = result
    write_json(base_perform,os.path.join(saved_root,'base_performance.json'))
    write_json(contagion.tolist(),os.path.join(saved_root,'contagion_matrix.json'))
    write_json(block_perform,os.path.join(saved_root,'block_performance.json'))
    write_gff(errors, os.path.join(saved_root,'error_status.gff'))
    write_json(site_p_a_diff,os.path.join(saved_root,'p_a_abs_diff.json'))
    write_json(site_a_p_abs_diff,os.path.join(saved_root,'a_p_abs_diff.json'))

def main(predict_path,answer_path,region_table_path,saved_root,**kwargs):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    region_table = read_region_table(region_table_path)
    compare_and_save(predict,answer,region_table,saved_root,**kwargs)

if __name__ == '__main__':
    parser = ArgumentParser(description='Compare predict GFF to answer GFF')
    parser.add_argument("--predict_path",help='The path of prediction result in GFF format',required=True)
    parser.add_argument("--answer_path",help='The path of answer result in GFF format',required=True)
    parser.add_argument("--region_table_path",help='The path of region table',required=True)
    parser.add_argument("--saved_root",help="Path to save result",required=True)
    
    args = parser.parse_args()
    create_folder(args.saved_root)
    config_path = os.path.join(args.saved_root,'performance_setting.json')
    config = vars(args)
    write_json(config,config_path)
    main(args.predict_path,args.answer_path,args.fai_path,args.saved_root)
