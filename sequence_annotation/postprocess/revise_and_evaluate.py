import os
import sys
import math
import torch
import numpy as np
import pandas as pd
from multiprocessing import Pool
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES, create_folder
from sequence_annotation.utils.utils import read_gff, read_json, write_gff, write_json
from sequence_annotation.genome_handler.select_data import load_data
from sequence_annotation.genome_handler.ann_genome_processor import simplify_genome
from sequence_annotation.preprocess.utils import get_data_names,read_region_table
from sequence_annotation.preprocess.length_gaussian_modeling import norm_fit_log10
from sequence_annotation.preprocess.utils import get_gff_with_intergenic_region
from sequence_annotation.process.seq_ann_engine import get_best_model_and_origin_executor
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine, get_batch_size
from sequence_annotation.process.performance import main as performance_main
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.postprocess.boundary_process import get_splicing_regex
from sequence_annotation.postprocess.path_helper import PathHelper
from sequence_annotation.postprocess.gff_reviser import main as gff_revise_main

def get_splicing_kwargs(path_helper, first_n=None):
    donor_regex = get_splicing_regex(
        path_helper.donor_signal_stats_path,
        first_n=first_n)
    acceptor_regex = get_splicing_regex(
        path_helper.acceptor_signal_stats_path, first_n=first_n)
    kwargs = {}
    kwargs['donor_pattern'] = donor_regex
    kwargs['acceptor_pattern'] = acceptor_regex
    kwargs['donor_index_shift'] = 0
    kwargs['acceptor_index_shift'] = 1
    return kwargs


def get_distances(root, scale):
    a_p_abs_diff = read_json(
        os.path.join(
            root,
            'a_p_abs_diff.json'))
    donor_distance = a_p_abs_diff['mean']['splicing_donor_site'] * scale
    acceptor_distance = a_p_abs_diff['mean']['splicing_acceptor_site'] * scale
    return {'donor_distance': donor_distance,
            'acceptor_distance': acceptor_distance}


def get_overall_loss(root):
    block_performance = read_json(os.path.join(root,'block_performance.json'))
    base_performance = read_json(os.path.join(root,'base_performance.json'))
    abs_diff = read_json(os.path.join(root,'abs_diff.json'))
    site_matched = read_json(os.path.join(root,'site_matched.json'))
    block_f1_keys = ['internal_exon_F1', 'exon_F1', 'intron_F1']
    block_chain_f1_keys = ['gene_F1', 'intron_chain_F1']
    base_f1_keys = ['F1_exon', 'F1_intron', 'F1_other', 'macro_F1']
    loss = 0
    count = 0
    for key in base_f1_keys:
        value = base_performance[key]
        loss += (1 - value)
        count += 1
    
    for key in block_f1_keys:
        value = block_performance[key]
        loss += (1 - value)
        count += 1
            
    for key in block_chain_f1_keys:
        value = block_performance[key]
        if str(value) == 'nan':
            value = 0
        loss += (1 - value)
        count += 1

    for value in abs_diff['mean'].values():
        loss += (1-1/(value + 1))
        #loss += np.log10(value + 1))
        count += 1

    for value in site_matched['F1'].values():
        loss += (1 - value)
        count += 1

    loss /= count
    return loss


def get_length_thresholds(length_gaussian_path, std_num):
    length_model = pd.read_csv(length_gaussian_path, sep='\t')
    length_thresholdss = {}
    length_model['threshold'] = length_model['mean'] - \
        length_model['std'] * std_num
    for type_, group in length_model.groupby('type'):
        threshold = min(group['threshold'])
        if threshold < 0:
            threshold = 0
        if type_ == 'gene':
            type_ = 'transcript'
        length_thresholdss[type_] = pow(10, threshold)
    length_thresholdss['other'] = 0
    return length_thresholdss


class Reviser:
    def __init__(self, plust_strand_gff_path, fasta_path, region_table_path,multiprocess=None):
        self._plus_strand_gff_path = plust_strand_gff_path
        self._fasta_path = fasta_path
        self._region_table_path = region_table_path
        self.multiprocess = multiprocess
        
    def revise(self, saved_root,**kwargs):
        gff_revise_main(saved_root, self._plus_strand_gff_path,
                        self._region_table_path,self._fasta_path,
                        revised_config_or_path=kwargs,
                       multiprocess=self.multiprocess)

class Evaluator:
    def __init__(self, answer_path, region_table_path):
        self._answer_path = answer_path
        self._region_table_path = region_table_path

    def evaluate(self, predict_path, saved_root):
        loss_path = os.path.join(saved_root, 'overall_loss.txt')
        performance_main(predict_path, self._answer_path,
                         self._region_table_path,saved_root)
        overall_loss = get_overall_loss(saved_root)
        with open(loss_path, "w") as fp:
            fp.write("{}\n".format(overall_loss))
        return overall_loss


class ReviseEvaluator:
    def __init__(self, path_helper, predicted_root,multiprocess=None):
        self.path_helper = path_helper
        self.predicted_root = predicted_root
        self.plus_strand_gff_path = os.path.join(predicted_root, 'test_predict_plus_strand.gff3')
        self._region_table_path = path_helper.region_table_path
        self._fasta_path = path_helper.fasta_path
        self._answer_path = path_helper.answer_path
        self.multiprocess = multiprocess
        self.reviser = Reviser(
            self.plus_strand_gff_path,
            self._fasta_path,
            self._region_table_path,
            multiprocess=multiprocess)
        self.evaluator = Evaluator(self._answer_path, self._region_table_path)

    def process(self, revised_root, **kwargs):
        revised = os.path.join(revised_root, 'revised_double_strand.gff3')
        self.reviser.revise(revised_root, **kwargs)
        return self.evaluator.evaluate(revised, revised_root)


def test(trained_root, test_result_root, path_helper):
    loss_path = os.path.join(test_result_root, 'overall_loss.txt')
    if True:#not os.path.exists(loss_path):
        data_path = path_helper.processed_data_path
        answer_path = path_helper.answer_path
        create_folder(test_result_root)
        best_model, executor = get_best_model_and_origin_executor(trained_root)
        batch_size = get_batch_size(trained_root)
        engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
        engine.batch_size = batch_size
        engine.set_root(test_result_root,with_train=False,
                        with_val=False,with_test=False,
                        create_tensorboard=False)
        region_table_path=path_helper.region_table_path
        singal_handler = engine.get_signal_handler(test_result_root,prefix='test',
                                                   inference=executor.inference,
                                                   region_table_path=region_table_path,
                                                   answer_gff_path=answer_path)
        callbacks = Callbacks()
        callbacks.add(singal_handler)
        #Set loader
        test_seqs, test_ann_seqs = load_data(data_path)
        raw_data = {'testing': {'inputs': test_seqs, 'answers': test_ann_seqs}}
        data = engine.process_data(raw_data)
        data_loader = engine.create_basic_data_gen()(data['testing'])
        engine.test(best_model, executor, data_loader,callbacks=callbacks)
        overall_loss = get_overall_loss(test_result_root)
        with open(loss_path, "w") as fp:
            fp.write("{}\n".format(overall_loss))
    else:
        with open(loss_path, "r") as fp:
            overall_loss = float(fp.read())
    return overall_loss


def get_intergenic_threshold(predicted_gff,region_table,chrom_id_source):
    gff = get_gff_with_intergenic_region(predicted_gff,region_table,chrom_id_source)
    intergenic_region = gff[gff['feature']=='intergenic region']
    lengths = list(intergenic_region['end']-intergenic_region['start']+1)
    params = norm_fit_log10(lengths, component_num=4)
    index = np.argmin(params['means'])
    threshold = math.pow(10,params['means'][index])
    return threshold

class RevierSpaceSearcher:
    def __init__(self,revise_evaluator,output_root):
        self._revise_evaluator = revise_evaluator
        self._path_helper = self._revise_evaluator.path_helper
        self._predicted_root = self._revise_evaluator.predicted_root
        self._output_root = output_root
        self._records = None
        self._index = None
        self._std_num = None
        self._length_thresholds = None
        self._first_n_motif = None
        self._methods_1ist = None
        self._init_revised_space()

    def _init_revised_space(self):
        self._std_num = 3
        self._distance_scales = [0,1,2,3]
        self._first_n_motif = 1
        gff = read_gff(self._revise_evaluator.plus_strand_gff_path)
        region_table = read_region_table(self._path_helper.region_table_path)
        intergenic_threshold = get_intergenic_threshold(gff,region_table,'ordinal_id_with_strand')
        self._length_thresholds = get_length_thresholds(self._path_helper.length_log10_model_path,self._std_num)
        self._length_thresholds['other'] = intergenic_threshold
        self._methods_1ist = [['length_threshold'],['distance_threshold'],
                              ['length_threshold', 'distance_threshold'],
                              ['distance_threshold', 'length_threshold']]

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['std_num'] = self._std_num
        config['distance_scales'] = self._distance_scales
        config['first_n_motif'] = self._first_n_motif
        config['methods_1ist'] = self._methods_1ist
        return config

    def _execute_revise_evaluator(self, target, methods, distances):
        print("Working on {}".format(target))
        revised_root = os.path.join(self._output_root,target)
        splicing_kwargs = get_splicing_kwargs(self._path_helper,
                                              first_n=self._first_n_motif)
        self._revise_evaluator.process(revised_root, methods=methods,
                                       length_thresholds=self._length_thresholds,
                                       acceptor_distance=distances['acceptor_distance'],
                                       donor_distance=distances['donor_distance'],
                                       **splicing_kwargs)
        print("Finish working on {}".format(target))

    def _revise_and_evaluate_on_val(self, methods,distance_scale=None):
        self._index += 1
        print("Revising {}".format(self._index))
        target = 'revised_val_{}'.format(self._index)
        hyperparams = {'distance_scale': distance_scale, 'methods': methods,'target': target}
        self._records.append(hyperparams)
        if distance_scale is not None:
            distances = get_distances(self._predicted_root,distance_scale)
        else:
            distances = {'acceptor_distance': None, 'donor_distance': None}
        kwargs = (target,methods,distances)
        return kwargs

    def search(self):
        best_hyperparams = {'distance_scale': None,'methods': None}
        self._index = 0
        self._records = []
        overall_loss_path = os.path.join(self._predicted_root,'overall_loss.txt')
        with open(overall_loss_path, "r") as fp:
            val_loss = float(fp.read())
        best_loss = val_loss
        best_target = 'source'
        best_hyperparams = {'distance_scale': None,'methods': None,'target':'source',
                            'loss':val_loss}
        self._records.append(best_hyperparams)

        kwargs_list = []
        for methods in self._methods_1ist:
            if 'distance_threshold' in methods:
                for distance_scale in self._distance_scales:                
                    kwargs = self._revise_and_evaluate_on_val(methods,
                                                              distance_scale=distance_scale)
                    kwargs_list.append(kwargs)
            else:
                kwargs = self._revise_and_evaluate_on_val(methods)
                kwargs_list.append(kwargs)

        with Pool(processes=40) as pool:
            pool.starmap(self._execute_revise_evaluator, kwargs_list)

        for record in self._records:
            if record['target'] != 'source':
                overall_loss_path = os.path.join(self._output_root,record['target'],'overall_loss.txt')
            with open(overall_loss_path, "r") as fp:
                loss = float(fp.read())
            record['loss'] = loss
            if best_loss > loss:
                best_hyperparams = record
                best_loss = loss
                best_target = record['target']

        record_path = os.path.join(self._output_root, 'record.tsv')
        record = pd.DataFrame.from_dict(self._records)
        record.to_csv(record_path,sep='\t',index=None)
        write_json(best_hyperparams,os.path.join(self._output_root,'best_revised_config.json'))
        if best_target!='source':
            gff_reviser_config_path = os.path.join(self._output_root,best_target,'reviser_config.json')
        else:
            gff_reviser_config_path = os.path.join(self._output_root,self._predicted_root,'reviser_config.json')
        gff_reviser_config = read_json(gff_reviser_config_path)
        write_json(gff_reviser_config,os.path.join(self._output_root,'best_gff_reviser_config.json'))
        return gff_reviser_config

def main(raw_data_root, trained_project_root, output_root,fold_name=None):
    
    processed_root = os.path.join(raw_data_root,'full_data')
    path_helper = PathHelper(raw_data_root, processed_root)
    data_names = get_data_names(path_helper.split_root)
    val_root = os.path.join(output_root, 'source')
    plus_gffs = []
    gffs = []
    create_folder(val_root)
    if fold_name is None:
        predicted_root = os.path.join(output_root, 'predicted')
        fold_names = list(data_names.keys())
        for trained_name in fold_names:
            for usage in ['validation', 'testing']:
                print("Predicted by {} for {}".format(trained_name,usage))
                trained_root = os.path.join(trained_project_root, trained_name)
                result_root = os.path.join(predicted_root, trained_name, usage)
                torch.cuda.empty_cache()
                path_helper = PathHelper(raw_data_root, processed_root,trained_name,usage)
                test(trained_root, result_root, path_helper)
                torch.cuda.empty_cache()

        for trained_name in fold_names:
            result_root = os.path.join(predicted_root, trained_name, 'validation')
            plus_gff = read_gff(os.path.join(result_root, 'test_predict_plus_strand.gff3'))
            gff = read_gff(os.path.join(result_root, 'test_predict_double_strand.gff3'))
            plus_gffs.append(plus_gff)
            gffs.append(gff)

        merged_plus_gff = pd.concat(plus_gffs)
        merged_gff = pd.concat(gffs)
        
        plus_predicted_path = os.path.join(val_root,'test_predict_plus_strand.gff3')
        predicted_path = os.path.join(val_root,'test_predict_double_strand.gff3')
        write_gff(merged_plus_gff, plus_predicted_path)
        write_gff(merged_gff, predicted_path)

        print("Evaluating the merged result")
        path_helper = PathHelper(raw_data_root, processed_root)
        evaluator = Evaluator(path_helper.answer_path,path_helper.region_table_path)
        evaluator.evaluate(predicted_path,val_root)
        val_name = path_helper.train_val_name
        
    else:
        gff = []
        plus_gff = []
        for usage in ['validation', 'testing']:
            result_root = os.path.join(output_root, usage)
            torch.cuda.empty_cache()
            path_helper = PathHelper(raw_data_root, processed_root,fold_name,usage)
            test(trained_project_root, result_root, path_helper)
            torch.cuda.empty_cache()

        result_root = os.path.join(output_root, 'validation')
        plus_input_gff = read_gff(os.path.join(result_root,'test_predict_plus_strand.gff3'))
        input_gff = read_gff(os.path.join(result_root,'test_predict_double_strand.gff3'))

        plus_predicted_path = os.path.join(val_root,'test_predict_plus_strand.gff3')
        predicted_path = os.path.join(val_root,'test_predict_double_strand.gff3')
        
        write_gff(plus_input_gff, plus_predicted_path)
        write_gff(input_gff, predicted_path)
        
        path_helper = PathHelper(raw_data_root, processed_root,fold_name,'validation')
        evaluator = Evaluator(path_helper.answer_path, path_helper.region_table_path)
        evaluator.evaluate(predicted_path,val_root)
            
    
    revise_evaluator = ReviseEvaluator(path_helper,val_root)

    revised_val_root = os.path.join(output_root,'revised_val')
    space_searcher = RevierSpaceSearcher(revise_evaluator,revised_val_root)
    best_reviser_config = space_searcher.search()
    write_json(space_searcher.get_config(),os.path.join(revised_val_root,'auto_revised_config.json'))
    
    
    if fold_name is None:
        fold_names = list(data_names.keys())
        for trained_name in fold_names:
            path_helper = PathHelper(raw_data_root, processed_root,trained_name,'testing')
            revised_root = os.path.join(output_root, 'revised_test',trained_name)
            val_root = os.path.join(predicted_root, trained_name, 'testing')
            revise_evaluator = ReviseEvaluator(path_helper,val_root,multiprocess=40)
            torch.cuda.empty_cache()
            loss = revise_evaluator.process(revised_root, **best_reviser_config)
            torch.cuda.empty_cache()
    else:
        path_helper = PathHelper(raw_data_root, processed_root,fold_name,'testing')
        revised_root = os.path.join(output_root, 'revised_test')
        val_root = os.path.join(output_root, 'testing')
        revise_evaluator = ReviseEvaluator(path_helper,val_root,multiprocess=40)
        torch.cuda.empty_cache()
        loss = revise_evaluator.process(revised_root, **best_reviser_config)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--raw_data_root", required=True,
                        help="Root of Arabidopsis processed data")
    parser.add_argument("-i", "--trained_project_root", required=True,
                        help='The root of trained project')
    parser.add_argument("-o", "--output_root", required=True,
                        help="The root to save result")
    parser.add_argument("-g", "--gpu_id", type=int,
                        default=0, help="GPU to used")
    parser.add_argument("--fold_name")
    args = parser.parse_args()
    kwargs = dict(vars(args))
    del kwargs['gpu_id']
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
