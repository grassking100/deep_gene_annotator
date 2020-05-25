import os
import sys
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
from argparse import ArgumentParser
from  matplotlib import pyplot as plt
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, write_gff, write_json, read_gff,get_gff_with_updated_attribute
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES,get_gff_with_attribute,get_gff_with_feature_coord
from sequence_annotation.genome_handler.sequence import AnnSequence, PLUS, MINUS
from sequence_annotation.genome_handler.ann_seq_processor import get_background,seq2vecs
from sequence_annotation.preprocess.utils import read_region_table,get_gff_with_intron
from sequence_annotation.preprocess.utils import EXON_TYPES, RNA_TYPES,INTRON_TYPES,GENE_TYPES
from sequence_annotation.preprocess.get_id_table import get_id_table,convert_id_table_to_dict
from sequence_annotation.process.metric import calculate_metric,contagion_matrix,categorical_metric
from sequence_annotation.process.site_analysis import get_all_site_diff,get_all_site_matched_ratio,plot_site_diff

pd.set_option('mode.chained_assignment', 'raise')

STRAND_CONVERT = {'+':PLUS,'-':MINUS}

def _normalize_matrix(matrix):
    normed_matrix = []
    for row in matrix:
        normed_matrix.append([item/sum(row) for item in row])
    return normed_matrix

def _group_type_by_parent(gff,group_types):
    gff = gff[gff['feature'].isin(group_types)]
    groups = gff.groupby('parent')
    return groups

def _set_has_intron_status(gff):
    has_intron_ids = set(gff[gff['feature'].isin(INTRON_TYPES)]['parent'])
    transcripts = gff[gff['feature'].isin(RNA_TYPES)]
    gff.loc[transcripts[transcripts['id'].isin(has_intron_ids)].index,'has_intron'] = True

def _set_is_internal_exon_status(gff):
    transcripts = gff[gff['feature'].isin(RNA_TYPES)]
    starts = dict(zip(transcripts['id'],transcripts['start']))
    ends = dict(zip(transcripts['id'],transcripts['end']))
    exons = gff[gff['feature'].isin(EXON_TYPES)]
    parent_start = exons['parent'].map(starts)
    parent_end = exons['parent'].map(ends)
    index = exons[(exons['start']!=parent_start) & (exons['end']!=parent_end)].index
    gff.loc[index,'is_internal_exon'] = True

def _compare_exon_intron_block(predict,answer,predict_id_convert,answer_id_convert):
    types = EXON_TYPES+INTRON_TYPES
    predict_blocks = predict[predict['feature'].isin(types)]
    answer_blocks = answer[answer['feature'].isin(types)]
    predict_feature_coord = set(predict_blocks['feature_coord'])
    answer_feature_coord = set(answer_blocks['feature_coord'])
    TP_ids = list(answer_feature_coord.intersection(predict_feature_coord))
    predict.loc[predict_blocks[predict_blocks['feature_coord'].isin(TP_ids)].index,'match'] = True
    answer.loc[answer_blocks[answer_blocks['feature_coord'].isin(TP_ids)].index,'match'] = True
    matched_predict = predict[(predict['match'])&(predict['feature'].isin(types))]
    matched_answer = answer[(answer['match'])&(answer['feature'].isin(types))]
    #Assume one freature coord with one block only
    predict_id_table = dict(zip(predict['id'],predict['parent'])) 
    answer_id_table = dict(zip(answer['id'],answer['parent'])) 
    predict_parents = dict(zip(matched_predict['feature_coord'],matched_predict['parent'].map(predict_id_table)))
    answer_parents = dict(zip(matched_answer['feature_coord'],matched_answer['parent'].map(answer_id_table)))
    predict.loc[matched_predict.index,'partner_gene'] = matched_predict['feature_coord'].map(answer_parents)
    answer.loc[matched_answer.index,'partner_gene'] = matched_answer['feature_coord'].map(predict_parents)

def _compare_internal_exon(predict,answer):
    predict_feature_coord = set(predict[predict['is_internal_exon']]['feature_coord'])
    answer_feature_coord = set(answer[answer['is_internal_exon']]['feature_coord'])
    TP_ids = answer_feature_coord.intersection(predict_feature_coord)
    predict.loc[predict['feature_coord'].isin(TP_ids),'internal_exon_match'] = True
    answer.loc[answer['feature_coord'].isin(TP_ids),'internal_exon_match'] = True
    
def _compare_gene(predict,answer):
    predict_transcript_groups = _group_type_by_parent(predict,RNA_TYPES)
    answer_transcript_groups = _group_type_by_parent(answer,RNA_TYPES)
    predict_id_groups = predict.groupby('id')
    answer_id_groups = answer.groupby('id')
    for predict_gene_id,predict_transcript_group in predict_transcript_groups:
        answer_gene_ids = set(predict_transcript_group['partner_gene'])
        answer_gene_ids = set(filter(lambda x:x!=None,answer_gene_ids))
        if len(answer_gene_ids)==1:
            answer_gene_id = list(answer_gene_ids)[0]
            answer_transcript_group = answer_transcript_groups.get_group(answer_gene_id)
            if predict_transcript_group['match'].all() and answer_transcript_group['match'].all():
                predict.loc[predict_id_groups.get_group(predict_gene_id).index,'match'] = True
                answer.loc[answer_id_groups.get_group(answer_gene_id).index,'match'] = True

def _compare_chain_block(predict,answer,types,set_matched_key,predict_id_convert,answer_id_convert):
    predict_groups = _group_type_by_parent(predict,types)
    answer_groups = _group_type_by_parent(answer,types)
    predict_id_groups = predict.groupby('id')
    answer_id_groups = answer.groupby('id')
    answer_parent_groups = answer.groupby('parent')
    for predict_transcript_id,predict_group in predict_groups:
        if len(predict_group)>0:
            answer_gene_ids = set(predict_group['partner_gene'])
            answer_gene_ids = set(filter(lambda x:x!=None,answer_gene_ids))
            if len(answer_gene_ids)==1:
                answer_gene_id = list(answer_gene_ids)[0]
                predict_index = predict_id_groups.get_group(predict_transcript_id).index
                answer_transcript_ids = answer_parent_groups.get_group(answer_gene_id)['id']
                for answer_transcript_id in answer_transcript_ids:
                    answer_group = answer_groups.get_group(answer_transcript_id)
                    #If all of its block is same, then it is matched
                    if set(predict_group['feature_coord'])==set(answer_group['feature_coord']):
                        answer_index = answer_id_groups.get_group(answer_transcript_id).index
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
        raise Exception("Inconsist between {}, got {}".format(type_name,
                                                              set(TP_predict_ids).symmetric_difference(TP_answer_ids)))
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
    partners = answer[(~answer['partner_gene'].isna())&(answer['feature'].isin(RNA_TYPES))][['id','partner_gene']]
    return performances,error_df,partners

def _create_ann_seq(chrom,length,strand):
    ann_seq = AnnSequence(['gene']+BASIC_GENE_ANN_TYPES,length)
    ann_seq.chromosome_id = chrom
    ann_seq.strand = strand
    ann_seq.id = '{}_{}'.format(chrom,strand)
    return ann_seq

def _gene_gff2vec(gff,chrom,strand,length):
    gff = gff[(gff['chr']==chrom) &( gff['strand']==strand)]
    strand = STRAND_CONVERT[strand]
    ann_seq = _create_ann_seq(chrom,length,strand)
    for block in gff[gff['feature']=='gene'].to_dict('record'):
        ann_seq.add_ann('gene',1,block['start']-1,block['end']-1)
    for block in gff[gff['feature']=='exon'].to_dict('record'):
        ann_seq.add_ann('exon',1,block['start']-1,block['end']-1)
    other = get_background(ann_seq,['gene'])
    ann_seq.set_ann('other',other)
    ann_seq.op_not_ann('intron','gene','exon')
    ann_seq = ann_seq.get_subseq(ann_types=BASIC_GENE_ANN_TYPES)
    vec = np.array([seq2vecs(ann_seq)]).transpose(0,2,1)
    return vec

def _calculate_metric(predict,answer,chrom,strand,length):
    predict_vec = _gene_gff2vec(predict,chrom,strand,length)
    answer_vec = _gene_gff2vec(answer,chrom,strand,length)
    mask = np.ones((1,length))
    categorical_metric_ = categorical_metric(predict_vec,answer_vec,mask)
    contagion_matrix_ = np.array(contagion_matrix(predict_vec,answer_vec,mask))
    return categorical_metric_,contagion_matrix_
    

def gff_performance(predict,answer,region_table,round_value=None,multiprocess=None):
    """
    The method would compare data between prediction and answer, and return the performance result
    
    Parameters
    ----------
    predict : pandas.DataFrame
        GFF dataframe of prediction
    answer : pandas.DataFrame
        GFF dataframe of answer
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
        table_regions.add("{}_{}".format(item['ordinal_id_wo_strand'],item['strand']))

    for answer_region in answer_regions:
        if answer_region not in table_regions:
            raise Exception(answer_region,sorted(list(table_regions)))

    print("Get intron data")
    pre_time = time.time()
    predict = predict[~predict['feature'].isin(INTRON_TYPES)]
    answer = answer[~answer['feature'].isin(INTRON_TYPES)]
    predict = get_gff_with_feature_coord(get_gff_with_intron(get_gff_with_attribute(predict)))
    answer = get_gff_with_feature_coord(get_gff_with_intron(get_gff_with_attribute(answer)))
    new_time = time.time()
    print("Time spend:{}".format(new_time-pre_time))
    pre_time = new_time
    result = {}
    label_num=len(BASIC_GENE_ANN_TYPES)
    metric = {}
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    metric['T'] = 0
    metric['F'] = 0
    for type_ in ['TPs','FPs','TNs','FNs']:
        metric[type_] = [0]*label_num
    result['contagion_matrix'] = np.array([[0]*label_num]*label_num)
    
    ordinal_id_wo_strands = region_table['ordinal_id_wo_strand']
    lengths = region_table['length']
    strands = region_table['strand']
    predict_coord = predict['chr'] + "_"+predict['strand']
    answer_coord = answer['chr'] + "_"+answer['strand']
    kwarg_list = []
    for index in range(len(region_table)):
        chrom = ordinal_id_wo_strands[index]
        length = lengths[index]
        strand = strands[index]
        kwarg_list.append((predict,answer,chrom,strand,length))

        
    if multiprocess is None:
        results = [_calculate_metric(*kwargs) for kwargs in kwarg_list]
    else:
        with Pool(processes=multiprocess) as pool:
            results = pool.starmap(_calculate_metric, kwarg_list)
    
    for item in results:
        categorical_metric_,contagion_matrix_ = item
        result['contagion_matrix'] += contagion_matrix_
        for type_ in ['TPs','FPs','TNs','FNs']:
            for index in range(label_num):
                metric[type_][index] += categorical_metric_[type_][index]
        metric['T'] += categorical_metric_['T']
        metric['F'] += categorical_metric_['F']
    result['contagion_matrix'] = result['contagion_matrix'].tolist()
    print("Calculate base performance")
    result['base_performance'] = calculate_metric(metric,label_names=BASIC_GENE_ANN_TYPES,round_value=round_value)
    new_time = time.time()
    print("Time spend:{}".format(new_time-pre_time))
    pre_time = new_time

    print("Calculate block performance")
    result['block_performance'],result['error_status'],result['partners'] = block_performance(predict,answer,round_value=round_value)
    new_time = time.time()
    print("Time spend:{}".format(new_time-pre_time))
    pre_time = new_time
    print("Calculate site distance")
    result['p_a_abs_diff'] = get_all_site_diff(answer,predict,round_value=round_value,multiprocess=multiprocess)
    result['a_p_abs_diff'] = get_all_site_diff(answer,predict,answer_as_ref=False,round_value=round_value,
                                              multiprocess=multiprocess)
    result['p_a_abs_diff_exclude_zero'] = get_all_site_diff(answer,predict,round_value=round_value,
                                                            include_zero=False,multiprocess=multiprocess)
    result['a_p_abs_diff_exclude_zero'] = get_all_site_diff(answer,predict,answer_as_ref=False,
                                                            round_value=round_value,include_zero=False,
                                                            multiprocess=multiprocess)
    result['abs_diff'] = {}
    result['abs_diff_exclude_zero'] = {}
    for method,dict_ in result['p_a_abs_diff'].items():
        result['abs_diff'][method] = {}
        for key,value in dict_.items():
            result['abs_diff'][method][key] = (result['a_p_abs_diff'][method][key] + value)/2
            
    for method,dict_ in result['p_a_abs_diff_exclude_zero'].items():
        result['abs_diff_exclude_zero'][method] = {}
        for key,value in dict_.items():
            result['abs_diff_exclude_zero'][method][key] = (result['a_p_abs_diff_exclude_zero'][method][key] + value)/2
            
    new_time = time.time()
    print("Time spend:{}".format(new_time-pre_time))
    pre_time = new_time
    print("Calculate site matched ratio")
    result['site_matched'] = get_all_site_matched_ratio(answer,predict,round_value=round_value)
    new_time = time.time()
    print("Time spend:{}".format(new_time-pre_time))
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

def compare_and_save(predict,answer,region_table,saved_root,round_value=None,multiprocess=None):
    create_folder(saved_root)
    result = gff_performance(predict,answer,region_table,round_value,multiprocess=multiprocess)
    for key,data in result.items():
        if key == 'error_status':
            if not result['error_status'].empty:
                write_gff(result['error_status'], os.path.join(saved_root,'error_status.gff'))
        elif key == 'partners':
            result['partners'].to_csv(os.path.join(saved_root,'partners.csv'),index=None)
        else:
            write_json(result[key],os.path.join(saved_root,'{}.json'.format(key)))
    plot_site_diff(predict,answer,saved_root)
    plot_site_diff(predict,answer,saved_root,answer_as_ref=False)

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
    parser.add_argument("--multiprocess",type=int,default=None)
    
    args = parser.parse_args()

    config_path = os.path.join(args.saved_root,'performance_setting.json')
    config = vars(args)
    create_folder(args.saved_root)
    write_json(config,config_path)
    main(**config)
