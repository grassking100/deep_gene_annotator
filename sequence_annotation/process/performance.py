import os
import sys
import numpy as np
from multiprocessing import Pool
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, write_json
from sequence_annotation.utils.metric import MetricCalculator,get_confusion_matrix,get_categorical_metric
from sequence_annotation.file_process.utils import BASIC_GENE_ANN_TYPES, MINUS, PLUS
from sequence_annotation.file_process.utils import get_gff_with_feature_coord,create_empty_gff
from sequence_annotation.file_process.utils import EXON_TYPE, GENE_TYPE,INTERGENIC_REGION_TYPE, INTRON_TYPE,TRANSCRIPT_TYPE
from sequence_annotation.file_process.utils import get_gff_with_intron,read_gff
from sequence_annotation.file_process.get_region_table import read_region_table
from sequence_annotation.file_process.site_analysis import get_all_site_diff,get_all_site_ratio,plot_site_diff
from sequence_annotation.genome_handler.sequence import AnnSequence
from sequence_annotation.genome_handler.ann_seq_processor import get_background,seq2vecs

CHROM_SOURCE = 'ordinal_id_wo_strand'
STRAND_CONVERT = {'+':PLUS,'-':MINUS}

def calculate_coord_performance(predict_coords,answer_coords,name,round_value=None):
    predict_coords = filter(lambda x: x is not None, predict_coords)
    answer_coords = filter(lambda x: x is not None, answer_coords)
    predict_coords = set(predict_coords)
    answer_coords = set(answer_coords)
    FP = len(predict_coords - answer_coords)
    FN = len(answer_coords - predict_coords)
    TP = len(answer_coords.intersection(predict_coords))
    calculator = MetricCalculator(1,label_names=[name],round_value=round_value,details=True,macro_F1=False)
    return calculator({'TP':[TP],'FP':[FP],'FN':[FN]})

def calculate_gene_performance(predict_id_table,answer_id_table,predict_transcript_coord,
                               answer_transcript_coord,round_value=None):
    predict_coord = set()
    answer_coord = set()
    for gene_id,tanscript_ids in predict_id_table.items():
        coords = [predict_transcript_coord[id_] for id_ in tanscript_ids]
        coords = filter(lambda x: x is not None, coords)
        predict_coord.add('_'.join(sorted(coords)))
        
    for gene_id,tanscript_ids in answer_id_table.items():
        coords = [answer_transcript_coord[id_] for id_ in tanscript_ids]
        coords = filter(lambda x: x is not None, coords)
        answer_coord.add('_'.join(sorted(coords)))
    performance = calculate_coord_performance(predict_coord,answer_coord,'gene',round_value=round_value)
    return performance

def get_chain_blocks_coord(parent_ids,blocks):
    coords = {}
    if len(blocks)==0:
        for id_ in parent_ids:
            coords[id_] = None
    else:
        groups = blocks.groupby('parent')
        for id_ in parent_ids:
            if id_ in groups.groups.keys():
                group = groups.get_group(id_)
                coords[id_] = '_'.join(sorted(group['feature_coord']))
            else:
                coords[id_] = None
    return coords

def get_gene_transcript_table(gff):
    table = {}
    if len(gff) > 0:
        transcripts = gff[gff['feature']==TRANSCRIPT_TYPE]
        for parent,id_ in zip(transcripts['parent'],transcripts['id']):
            if parent not in table:
                table[parent] = []
            table[parent].append(id_)
    return table

def calculate_block_performance(predict,answer,round_value=None):
    if 'feature_coord' not in predict.columns:
        predict = get_gff_with_feature_coord(predict)
    if 'feature_coord' not in answer.columns:
        answer = get_gff_with_feature_coord(answer)
    performance = {}
    predict_id_table = get_gene_transcript_table(predict)
    answer_id_table = get_gene_transcript_table(answer)
    answer_transcript_ids = []
    predict_transcript_ids = []
    for ids in list(answer_id_table.values()):
        answer_transcript_ids += ids
    for ids in list(predict_id_table.values()):
        predict_transcript_ids += ids

    predict_exons = predict[predict['feature']==EXON_TYPE]
    answer_exons = answer[answer['feature']==EXON_TYPE]
    predict_introns = predict[predict['feature']==INTRON_TYPE]
    answer_introns = answer[answer['feature']==INTRON_TYPE]
    predicted_exon_coords = predict_exons['feature_coord']
    answer_exon_coords = answer_exons['feature_coord']
    predicted_intron_coords = predict_introns['feature_coord']
    answer_intron_coords = answer_introns['feature_coord']
    predicted_chained_exon_coords = get_chain_blocks_coord(predict_transcript_ids,predict_exons)
    answer_chained_exon_coords = get_chain_blocks_coord(answer_transcript_ids,answer_exons)
    predicted_chained_intron_coords = get_chain_blocks_coord(predict_transcript_ids,predict_introns)
    answer_chained_intron_coords = get_chain_blocks_coord(answer_transcript_ids,answer_introns)
    
    performance.update(calculate_coord_performance(predicted_exon_coords,answer_exon_coords,
                                                   'exon_block',round_value=round_value))
    performance.update(calculate_coord_performance(predicted_intron_coords,answer_intron_coords,
                                                   'intron_block',round_value=round_value))
    performance.update(calculate_coord_performance(predicted_chained_exon_coords.values(),
                                                   answer_chained_exon_coords.values(),
                                                   'transcript',round_value=round_value))
    performance.update(calculate_coord_performance(predicted_chained_intron_coords.values(),
                                                   answer_chained_intron_coords.values(),
                                                   'chained_introns',round_value=round_value))
    performance.update(calculate_gene_performance(predict_id_table,answer_id_table,
                                                  predicted_chained_exon_coords,
                                                  answer_chained_exon_coords,
                                                  round_value=round_value))
    return performance

def _gene_gff2vec(gff,chrom,strand,length):
    strand = STRAND_CONVERT[strand]
    ann_seq = AnnSequence([GENE_TYPE]+BASIC_GENE_ANN_TYPES,length)
    ann_seq.chromosome_id = chrom
    ann_seq.strand = strand
    ann_seq.id = '{}_{}'.format(chrom,strand)
    for index,block in gff[gff['feature']==GENE_TYPE].iterrows():
        ann_seq.add_ann(GENE_TYPE,1,block['start']-1,block['end']-1)
    for index,block in gff[gff['feature']==EXON_TYPE].iterrows():
        ann_seq.add_ann(EXON_TYPE,1,block['start']-1,block['end']-1)
    other = get_background(ann_seq,[GENE_TYPE])
    ann_seq.set_ann(INTERGENIC_REGION_TYPE,other)
    ann_seq.op_not_ann(INTRON_TYPE,GENE_TYPE,EXON_TYPE)
    vec = np.array([seq2vecs(ann_seq,BASIC_GENE_ANN_TYPES)]).transpose(0,2,1)
    return vec

def _calculate_metric(predict,answer,chrom,strand,length):
    predict_vec = _gene_gff2vec(predict,chrom,strand,length)
    answer_vec = _gene_gff2vec(answer,chrom,strand,length)
    mask = np.ones((1,length))
    categorical_metric = get_categorical_metric(predict_vec,answer_vec,mask)
    confusion_matrix = np.array(get_confusion_matrix(predict_vec,answer_vec,mask))
    return categorical_metric,confusion_matrix
    
def calculate_base_performance(predict,answer,region_table,round_value=None,multiprocess=None):
    predict = predict.copy()
    answer = answer.copy()
    predict['chr_strand'] = predict['chr'] +"_"+ predict['strand']
    answer['chr_strand'] = answer['chr'] +"_"+ answer['strand']
    predict_groups = predict.groupby('chr_strand')
    answer_groups = answer.groupby('chr_strand')
    label_num=len(BASIC_GENE_ANN_TYPES)
    result = {}
    metric = {}
    for type_ in ['TP','FP','TN','FN']:
        metric[type_] = [0]*label_num
    kwarg_list = []
    chroms = region_table[CHROM_SOURCE]
    lengths = region_table['length']
    strands = region_table['strand']
    for chrom,length,strand in zip(chroms,lengths,strands):
        chrom_strand = chrom +"_"+strand
        if chrom_strand in predict_groups.groups.keys():
            part_predict = predict_groups.get_group(chrom_strand)
        else:
            part_predict = create_empty_gff()
        if chrom_strand in answer_groups.groups.keys():
            part_answer = answer_groups.get_group(chrom_strand)
        else:
            part_answer = create_empty_gff()
        kwarg_list.append((part_predict,part_answer,chrom,strand,length))

    if multiprocess is None:
        results = [_calculate_metric(*kwargs) for kwargs in kwarg_list]
    else:
        with Pool(processes=multiprocess) as pool:
            results = pool.starmap(_calculate_metric, kwarg_list)
    result['contagion_matrix'] = np.array([[0]*label_num]*label_num)
    for item in results:
        categorical_metric_,contagion_matrix_ = item
        result['contagion_matrix'] += contagion_matrix_
        for type_ in ['TP','FP','FN']:
            for index in range(label_num):
                metric[type_][index] += categorical_metric_[type_][index]
    result['contagion_matrix'] = result['contagion_matrix'].tolist()
    calculator = MetricCalculator(len(BASIC_GENE_ANN_TYPES),label_names=BASIC_GENE_ANN_TYPES,round_value=round_value)
    result['categorical_metric'] = calculator(metric)
    return result
    
def calculate_site_distance(answer,predict,round_value=None,multiprocess=None):
    result = {}
    result['p_a_abs_diff'] = get_all_site_diff(answer,predict,round_value=round_value,
                                               multiprocess=multiprocess)
    result['a_p_abs_diff'] = get_all_site_diff(answer,predict,answer_as_ref=False,
                                               round_value=round_value,multiprocess=multiprocess)
    result['abs_diff'] = {'mean':{}}
    if 'mean' in result['p_a_abs_diff'] and 'mean' in result['a_p_abs_diff']:
        for key,value in result['p_a_abs_diff']['mean'].items():
            result['abs_diff']['mean'][key] = (result['a_p_abs_diff']['mean'][key] + value)/2
    return result
    
def gff_performance(predict,answer,region_table,
                    round_value=None,multiprocess=None,calculate_base=True):
    """
    The method would compare data between prediction and answer, and return the performance result
    
    Parameters
    ----------
    predict : pandas.DataFrame
        GFF dataframe of prediction on double-strand data
    answer : pandas.DataFrame
        GFF dataframe of answer on double-strand data
    round_value : int, optional
        rounded at the specified number of digits, if it is None then the result wouldn't be rounded (default is None)
    Returns
    -------
    result : dict
        The dictionary about performance
    """
    chroms = set(region_table[CHROM_SOURCE]+region_table['strand'])
    invalid_predict_chroms = set(predict['chr']+predict['strand']) - chroms
    invalid_answer_chroms = set(answer['chr']+answer['strand']) - chroms
    if len(invalid_predict_chroms)!=0:
        raise Exception("Invalid chrom {} in predicted data".format(invalid_predict_chroms))
    if len(invalid_answer_chroms)!=0:
        raise Exception("Invalid chrom {} in answer data".format(invalid_answer_chroms))
    predict = get_gff_with_intron(predict)
    answer = get_gff_with_intron(answer)
    result = {}
    if calculate_base:
        print("Calculate base performance")
        result['base_performance'] = calculate_base_performance(predict,answer,region_table,
                                                                round_value=round_value,
                                                                multiprocess=multiprocess)
    print("Calculate block performance")
    result['block_performance'] = calculate_block_performance(predict,answer,round_value=round_value)
    print("Calculate distance performance")
    result['distance'] = calculate_site_distance(answer,predict,round_value=round_value)
    print("Calculate site performance")
    result['site_matched'] = get_all_site_ratio(answer,predict,round_value=round_value)
    return result

def compare_and_save(predict,answer,region_table,output_root,**kwargs):
    create_folder(output_root)
    result = gff_performance(predict,answer,region_table,**kwargs)
    for key,data in result.items():
        write_json(result[key],os.path.join(output_root,'{}.json'.format(key)))
    plot_site_diff(predict,answer,output_root)
    plot_site_diff(predict,answer,output_root,answer_as_ref=False)

def main(predict_path,answer_path,region_table_path,output_root,
         multiprocess=None,calculate_base=True):
    predict = read_gff(predict_path)
    answer = read_gff(answer_path)
    region_table = read_region_table(region_table_path)
    compare_and_save(predict,answer,region_table,output_root,multiprocess=multiprocess,
                     calculate_base=calculate_base)

if __name__ == '__main__':
    parser = ArgumentParser(description='Compare predict GFF to answer GFF')
    parser.add_argument("-p","--predict_path",help='The path of prediction result in GFF format',required=True)
    parser.add_argument("-a","--answer_path",help='The path of answer result in GFF format',required=True)
    parser.add_argument("-r","--region_table_path",help='The path of region table',required=True)
    parser.add_argument("-s","--output_root",help="Path to save result",required=True)
    parser.add_argument("--multiprocess",type=int,default=None)
    
    args = parser.parse_args()

    config_path = os.path.join(args.output_root,'performance_setting.json')
    config = vars(args)
    create_folder(args.output_root)
    write_json(config,config_path)
    main(**config)
