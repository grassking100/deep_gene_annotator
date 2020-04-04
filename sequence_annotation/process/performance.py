import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from  matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, write_gff, write_json, read_gff,get_gff_with_updated_attribute
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES,get_gff_with_attribute,get_gff_with_feature_coord
from sequence_annotation.genome_handler.sequence import AnnSequence, PLUS, MINUS
from sequence_annotation.genome_handler.ann_seq_processor import get_background,seq2vecs
from sequence_annotation.preprocess.utils import read_region_table
from sequence_annotation.preprocess.utils import EXON_TYPES, RNA_TYPES,INTRON_TYPES,GENE_TYPES
from sequence_annotation.preprocess.utils import get_gff_with_intron
from sequence_annotation.preprocess.get_id_table import get_id_table,convert_id_table_to_dict
from sequence_annotation.process.metric import calculate_metric,contagion_matrix,categorical_metric
from sequence_annotation.process.site_analysis import get_all_site_diff,get_all_site_matched_ratio

pd.set_option('mode.chained_assignment', 'raise')

STRAND_CONVERT = {'+':PLUS,'-':MINUS}

def _normalize_matrix(matrix):
    normed_matrix = []
    for row in matrix:
        normed_matrix.append([item/sum(row) for item in row])
    return normed_matrix

def _get_group_dict_by_transcript(gff,group_types):
    group = {}
    transcript_ids = gff[gff['feature'].isin(RNA_TYPES)]['id']
    for transcript_id in transcript_ids:
        group_ = gff[(gff['feature'].isin(group_types)) & (gff['parent']==transcript_id)]
        group[transcript_id] = group_
    return group

def _get_transcript_group_dict_by_gene(gff):
    group = {}
    gene_ids = gff[gff['feature'].isin(GENE_TYPES)]['id']
    for gene_id in gene_ids:
        group_ = gff[(gff['feature'].isin(RNA_TYPES)) & (gff['parent']==gene_id)]
        group[gene_id] = group_
    return group

def _set_has_intron_status(gff):
    transcript_ids = set(gff[gff['feature'].isin(RNA_TYPES)]['id'])
    for transcript_id in transcript_ids:
        group_ = gff[(gff['feature'].isin(INTRON_TYPES)) & (gff['parent']==transcript_id)]
        gff.loc[gff['id']==transcript_id,'has_intron'] = len(group_)>0

def _set_is_internal_exon_status(gff):
    exon_group_dict = _get_group_dict_by_transcript(gff,EXON_TYPES)
    for transcript_id,exon_group in exon_group_dict.items():
        transcript = gff[gff['id']==transcript_id].to_dict('record')[0]
        index = exon_group[(exon_group['start']!=transcript['start']) & (exon_group['end']!=transcript['end'])].index
        gff.loc[index,'is_internal_exon'] = True

def _compare_exon_intron_block(predict,answer,predict_id_convert,answer_id_convert):
    types = EXON_TYPES+INTRON_TYPES
    predict_exons = predict[predict['feature'].isin(types)]
    answer_exons = answer[answer['feature'].isin(types)]
    predict_feature_coord = set(predict_exons['feature_coord'])
    answer_feature_coord = set(answer_exons['feature_coord'])
    TP_ids = list(answer_feature_coord.intersection(predict_feature_coord))
    predict_groups = predict_exons.groupby('feature_coord')
    answer_groups = answer_exons.groupby('feature_coord')
    for id_ in TP_ids:
        predict_group = predict_groups.get_group(id_)
        answer_group = answer_groups.get_group(id_)
        answer_partner_transcript = list(predict.loc[predict_group.index,'parent'])[0]
        predict_partner_transcript = list(answer.loc[answer_group.index,'parent'])[0]
        predict.loc[predict_group.index,'match'] = True
        answer.loc[answer_group.index,'match'] = True
        answer.loc[answer_group.index,'partner_gene'] = predict_id_convert[answer_partner_transcript]
        predict.loc[predict_group.index,'partner_gene'] = answer_id_convert[predict_partner_transcript]

def _compare_internal_exon(predict,answer):
    predict_feature_coord = set(predict[predict['is_internal_exon']]['feature_coord'])
    answer_feature_coord = set(answer[answer['is_internal_exon']]['feature_coord'])
    TP_ids = answer_feature_coord.intersection(predict_feature_coord)
    predict.loc[predict['feature_coord'].isin(TP_ids),'internal_exon_match'] = True
    answer.loc[answer['feature_coord'].isin(TP_ids),'internal_exon_match'] = True
    
def _compare_gene(predict,answer):
    predict_transcript_group_dict = _get_transcript_group_dict_by_gene(predict)
    answer_transcript_group_dict = _get_transcript_group_dict_by_gene(answer)
    for predict_gene_id,predict_transcript_group in predict_transcript_group_dict.items():
        answer_gene_ids = set(predict_transcript_group['partner_gene'])
        answer_gene_ids = set(filter(lambda x:x!=None,answer_gene_ids))
        if len(answer_gene_ids)==1:
            answer_gene_id = list(answer_gene_ids)[0]
            answer_transcript_group = answer_transcript_group_dict[answer_gene_id]
            if predict_transcript_group['match'].all() and answer_transcript_group['match'].all():
                predict.loc[predict['id']==predict_gene_id,'match'] = True
                answer.loc[answer['id']==answer_gene_id,'match'] = True

def _compare_chain_block(predict,answer,types,set_matched_key,predict_id_convert,answer_id_convert):
    predict_group_dict = _get_group_dict_by_transcript(predict,types)
    answer_group_dict = _get_group_dict_by_transcript(answer,types)
    for predict_transcript_id,predict_group in predict_group_dict.items():
        if len(predict_group)>0:
            answer_gene_ids = set(predict_group['partner_gene'])
            answer_gene_ids = set(filter(lambda x:x!=None,answer_gene_ids))
            if len(answer_gene_ids)==1:
                answer_gene_id = list(answer_gene_ids)[0]
                predict_index = predict[predict['id']==predict_transcript_id].index
                answer_transcript_ids = answer[answer['parent'] == answer_gene_id]['id']
                for answer_transcript_id in answer_transcript_ids:
                    answer_group = answer_group_dict[answer_transcript_id]
                    #If all of its block is same, then it is matched
                    if set(predict_group['feature_coord'])==set(answer_group['feature_coord']):
                        answer_index = answer[answer['id']==answer_transcript_id].index
                        predict.loc[predict_index,set_matched_key] = True
                        answer.loc[answer_index,set_matched_key] = True
                        predict.loc[predict_index,'partner_gene'] = answer_id_convert[answer_transcript_id]
                        answer.loc[answer_index,'partner_gene'] = predict_id_convert[predict_transcript_id]

def _get_error(id_name,type_name,predict,answer,FP_ids,FN_ids):
    FP_data = predict[predict[id_name].isin(FP_ids)].copy()
    FN_data = answer[answer[id_name].isin(FN_ids)].copy()
    FP_data['feature']='wrong_{}'.format(type_name)
    FN_data['feature']='missing_{}'.format(type_name)
    error_df = pd.concat([FP_data,FN_data],sort=True)
    return error_df

def _get_status(predict,answer,id_name,query_key=None):
    query_key = query_key or 'match'
    predict_ids = set(predict[id_name])
    answer_ids = set(answer[id_name])
    TP_predict_ids = set(predict[predict[query_key]][id_name])
    TP_answer_ids = set(answer[answer[query_key]][id_name])
    FP_ids = predict_ids-TP_predict_ids
    FN_ids = answer_ids-TP_answer_ids
    ids = {'predict_ids':predict_ids,
           'answer_ids':answer_ids,
           'TP_predict_ids':TP_predict_ids,
           'TP_answer_ids':TP_answer_ids,
           'FP_ids':FP_ids,
           'FN_ids':FN_ids}
    return ids

def _calculate(TP,FP,FN,answer_num,predict_num,type_name,round_value=None):
    if (TP+FN)!=answer_num or (TP+FP)!=predict_num:
        raise Exception("Wrong number in {}".format(type_name))
        
    recall = float('nan')
    precision = float('nan')
    F1=float('nan')
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

def _get_performance(type_name,predict_ids,answer_ids,TP_predict_ids,TP_answer_ids,
                     FP_ids,FN_ids,round_value=None):

    if len(TP_predict_ids) != len(TP_answer_ids):
        raise Exception("Inconsist number at {}, got {} and {}".format(type_name,
                                                                       TP_predict_ids,
                                                                       TP_answer_ids))
    predict_num = len(predict_ids)
    answer_num = len(answer_ids)
    TP = len(TP_predict_ids)
    FP = len(FP_ids)
    FN = len(FN_ids)
    status = _calculate(TP,FP,FN,answer_num,predict_num,type_name,round_value=round_value)
    return status

def block_performance(predict,answer,round_value=None):
    if 'feature_coord' not in predict.columns:
        raise Exception("the predict file lacks 'feature_coord' column")
    if 'feature_coord' not in answer.columns:
        raise Exception("the answer file lacks 'feature_coord' column")

    for gff in [answer,predict]:
        gff['partner_gene']=None
        gff['match']=False
        gff['has_intron']=False
        gff['intron_chain_match']=False
        gff['is_internal_exon']=False
        gff['internal_exon_match']=False
        _set_is_internal_exon_status(gff)
        _set_has_intron_status(gff)
        
    predict_id_convert = convert_id_table_to_dict(get_id_table(predict))
    answer_id_convert = convert_id_table_to_dict(get_id_table(answer))
        
    _compare_exon_intron_block(predict,answer,predict_id_convert,answer_id_convert)
    _compare_internal_exon(predict,answer)
    _compare_chain_block(predict,answer,EXON_TYPES,'match',predict_id_convert,answer_id_convert)
    _compare_chain_block(predict,answer,INTRON_TYPES,'intron_chain_match',predict_id_convert,answer_id_convert)
    #Get introns
    predict_introns = predict[predict['feature'].isin(INTRON_TYPES)]
    answer_introns = answer[answer['feature'].isin(INTRON_TYPES)]
    #Get exons
    predict_exons = predict[predict['feature'].isin(EXON_TYPES)]
    answer_exons = answer[answer['feature'].isin(EXON_TYPES)]
    #Get transcripts
    predict_transcripts = predict[predict['feature'].isin(RNA_TYPES)]

    answer_transcripts = answer[answer['feature'].isin(RNA_TYPES)]
    #Get intron chains
    predict_intron_chains = predict[predict['feature'].isin(RNA_TYPES) & (predict['has_intron'])]
    answer_intron_chains = answer[answer['feature'].isin(RNA_TYPES) & (answer['has_intron'])]
    #Get internal exons
    predict_internal_exons = predict[predict['feature'].isin(EXON_TYPES) & (predict['is_internal_exon'])]
    answer_internal_exons = answer[answer['feature'].isin(EXON_TYPES) & (answer['is_internal_exon'])]
    
    #Get classification id status
    intron_status = _get_status(predict_introns,answer_introns,'feature_coord')
    exon_status = _get_status(predict_exons,answer_exons,'feature_coord')
    transcript_status = _get_status(predict_transcripts,answer_transcripts,'id')
    intron_chain_status = _get_status(predict_intron_chains,answer_intron_chains,'id',query_key='intron_chain_match')
    internal_exon_status = _get_status(predict_internal_exons,answer_internal_exons,'feature_coord',query_key='internal_exon_match')

    #Compare gene and get status
    _compare_gene(predict,answer)
    predict_genes = predict[predict['feature'].isin(GENE_TYPES)]
    answer_genes = answer[answer['feature'].isin(GENE_TYPES)]
    gene_status = _get_status(predict_genes,answer_genes,'id')
    status_list = [intron_status,exon_status,transcript_status,
                   intron_chain_status,internal_exon_status,gene_status]
    #Set type name and id names
    type_names = ['intron','exon','transcript','intron_chain','internal_exon','gene']
    id_names = ['feature_coord','feature_coord','id','id','feature_coord','id']
    #Get error df and performance
    error_list = []
    performances = {}
    for id_name,type_name,status in zip(id_names,type_names,status_list):
        df = _get_error(id_name,type_name,predict,answer,status['FP_ids'],status['FN_ids'])
        performance = _get_performance(type_name,**status,round_value=round_value)
        error_list.append(df)
        performances.update(performance)
    error_df = pd.concat(error_list,ignore_index=True)
    error_df = get_gff_with_updated_attribute(error_df)
    return performances,error_df

def _create_ann_seq(chrom,length,strand):
    ann_seq = AnnSequence(['gene']+BASIC_GENE_ANN_TYPES,length)
    ann_seq.chromosome_id = chrom
    ann_seq.strand = strand
    ann_seq.id = '{}_{}'.format(chrom,strand)
    return ann_seq

def _gene_gff2vec(gff,chrom_id,strand,length):
    subgff = gff[(gff['chr']==chrom_id) & (gff['strand']==strand)]
    strand = STRAND_CONVERT[strand]
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

def gff_performance(predict,answer,region_table,chrom_target,round_value=None):
    """
    The method would compare data between prediction and answer, and return the performance result
    
    Parameters
    ----------
    predict : pandas.DataFrame
        GFF dataframe of prediction
    answer : pandas.DataFrame
        GFF dataframe of answer
    chrom_target : str
        Chromosome target in region_table
    round_value : int, optional
        rounded at the specified number of digits, if it is None then the result wouldn't be rounded (default is None)
    Returns
    -------
    result : dict
        The dictionary about performance
    """
    answer_list = answer.to_dict('record')
    region_table_list = region_table.to_dict('record')
    answer_regions = set()
    table_regions = set()
    for item in answer_list:
        answer_regions.add("{}_{}".format(item['chr'],item['strand']))
        
    for item in region_table_list:
        table_regions.add("{}_{}".format(item[chrom_target],item['strand']))

    for answer_region in answer_regions:
        if answer_region not in table_regions:
            raise Exception(answer_region,sorted(list(table_regions)))

    predict = predict[~predict['feature'].isin(INTRON_TYPES)]
    answer = answer[~answer['feature'].isin(INTRON_TYPES)]
    predict = get_gff_with_feature_coord(get_gff_with_intron(get_gff_with_attribute(predict)))
    answer = get_gff_with_feature_coord(get_gff_with_intron(get_gff_with_attribute(answer)))
    result = {}
    result['block_performance'],result['error_status'] = block_performance(predict,answer,round_value=round_value)
    result['p_a_abs_diff'] = get_all_site_diff(answer,predict,round_value=round_value)
    result['a_p_abs_diff'] = get_all_site_diff(answer,predict,answer_as_ref=False,round_value=round_value)
    
    result['p_a_abs_diff_exclude_zero'] = get_all_site_diff(answer,predict,
                                                            round_value=round_value,include_zero=False)
    result['a_p_abs_diff_exclude_zero'] = get_all_site_diff(answer,predict,answer_as_ref=False,
                                                            round_value=round_value,include_zero=False)
    
    result['site_matched'] = get_all_site_matched_ratio(answer,predict,round_value=round_value)
    
    label_num=len(BASIC_GENE_ANN_TYPES)
    metric = {}
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    metric['T'] = 0
    metric['F'] = 0
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    result['contagion_matrix'] = np.array([[0]*label_num]*label_num)
    for item in region_table.to_dict('record'):
        chrom_id = item[chrom_target]
        length = item['length']
        strand = item['strand']
        predict_vec = _gene_gff2vec(predict,chrom_id,strand,length)
        answer_vec = _gene_gff2vec(answer,chrom_id,strand,length)
        mask = np.ones((1,length))
        metric_ = categorical_metric(predict_vec,answer_vec,mask)
        result['contagion_matrix'] += np.array(contagion_matrix(predict_vec,answer_vec,mask))
        for type_ in ['TPs','FPs','TNs','FNs']:
            for index in range(label_num):
                metric[type_][index] += metric_[type_][index]
        metric['T'] += metric_['T']
        metric['F'] += metric_['F']
    result['base_performance'] = calculate_metric(metric,label_names=BASIC_GENE_ANN_TYPES,round_value=round_value)
    return result

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

def compare_and_save(predict,answer,region_table,saved_root,chrom_target,round_value=None):
    create_folder(saved_root)
    result = gff_performance(predict,answer,region_table,chrom_target,round_value)
    write_json(result['base_performance'],os.path.join(saved_root,'base_performance.json'))
    write_json(result['contagion_matrix'].tolist(),os.path.join(saved_root,'contagion_matrix.json'))
    write_json(result['block_performance'],os.path.join(saved_root,'block_performance.json'))
    if not result['error_status'].empty:
        write_gff(result['error_status'], os.path.join(saved_root,'error_status.gff'))
    write_json(result['site_matched'],os.path.join(saved_root,'site_matched.json'))
    write_json(result['p_a_abs_diff'],os.path.join(saved_root,'p_a_abs_diff.json'))
    write_json(result['a_p_abs_diff'],os.path.join(saved_root,'a_p_abs_diff.json'))
    write_json(result['p_a_abs_diff_exclude_zero'],os.path.join(saved_root,'p_a_abs_diff_exclude_zero.json'))
    write_json(result['a_p_abs_diff_exclude_zero'],os.path.join(saved_root,'a_p_abs_diff_exclude_zero.json'))

def main(predict_path,answer_path,region_table_path,saved_root,**kwargs):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    region_table = read_region_table(region_table_path)
    compare_and_save(predict,answer,region_table,saved_root,**kwargs)

if __name__ == '__main__':
    parser = ArgumentParser(description='Compare predict GFF to answer GFF')
    parser.add_argument("-p","--predict_path",help='The path of prediction result in GFF format',required=True)
    parser.add_argument("-a","--answer_path",help='The path of answer result in GFF format',required=True)
    parser.add_argument("-r","--region_table_path",help='The path of region table',required=True)
    parser.add_argument("-s","--saved_root",help="Path to save result",required=True)
    parser.add_argument("-t","--chrom_target",help="Valid options are old_id and new_id",required=True)
    args = parser.parse_args()

    config_path = os.path.join(args.saved_root,'performance_setting.json')
    config = vars(args)
    create_folder(args.saved_root)
    write_json(config,config_path)
    main(**config)
