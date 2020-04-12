import os
import sys
import numpy as np
import pandas as pd
import torch
from multiprocessing import Process
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__) + "/../..")
from sequence_annotation.postprocess.boundary_process import get_splicing_regex
from sequence_annotation.postprocess.path_helper import PathHelper
from sequence_annotation.postprocess.gff_reviser import main as gff_revise_main
from sequence_annotation.process.seq_ann_engine import get_best_model_and_origin_executor, SeqAnnEngine, get_batch_size
from sequence_annotation.preprocess.rename_chrom import main as rename_chrom_main
from sequence_annotation.preprocess.utils import get_data_names
from sequence_annotation.process.performance import main as performance_main
from sequence_annotation.genome_handler.select_data import load_data
from sequence_annotation.utils.utils import BASIC_GENE_ANN_TYPES, create_folder, read_gff, read_json, write_gff, write_json

def get_splicing_kwargs(path_helper, first_n=None):
    donor_regex = get_splicing_regex(
        path_helper.donor_signal_stats_path,
        first_n=first_n)
    acceptor_regex = get_splicing_regex(
        path_helper.acceptor_signal_stats_path, first_n=first_n)
    kwargs = {}
    kwargs['donor_pattern'] = donor_regex
    kwargs['acceptor_pattern'] = acceptor_regex
    kwargs['donor_index_shift'] = 1
    kwargs['acceptor_index_shift'] = 2
    return kwargs


def get_distances(peformance_root, scale):
    a_p_abs_diff = read_json(
        os.path.join(
            peformance_root,
            'a_p_abs_diff_exclude_zero.json'))
    donor_distance = a_p_abs_diff['mean']['donor_site'] * scale
    acceptor_distance = a_p_abs_diff['mean']['acceptor_site'] * scale
    return {'donor_distance': donor_distance,
            'acceptor_distance': acceptor_distance}


def get_overall_loss(peformance_root):
    block_performance = read_json(
        os.path.join(
            peformance_root,
            'block_performance.json'))
    base_performance = read_json(
        os.path.join(
            peformance_root,
            'base_performance.json'))
    a_p_abs_diff = read_json(
        os.path.join(
            peformance_root,
            'a_p_abs_diff.json'))
    p_a_abs_diff = read_json(
        os.path.join(
            peformance_root,
            'p_a_abs_diff.json'))
    site_matched = read_json(
        os.path.join(
            peformance_root,
            'site_matched.json'))
    block_f1_keys = ['internal_exon_F1', 'exon_F1', 'intron_F1']
    block_chain_f1_keys = ['gene_F1', 'intron_chain_F1']
    base_loss = 1 - base_performance['macro_F1']
    block_loss = 0
    block_chain_loss = 0
    site_loss = 0
    matched_loss = 0
    for key, value in block_performance.items():
        if key in block_f1_keys:
            if str(value) == 'nan':
                value = 0
            block_loss += (1 - value)
        elif key in block_chain_f1_keys:
            if str(value) == 'nan':
                value = 0
            block_chain_loss += (1 - value)

    for value in a_p_abs_diff['mean'].values():
        site_loss += np.log10(value + 1)
    for value in p_a_abs_diff['mean'].values():
        site_loss += np.log10(value + 1)
    for value in site_matched['F1'].values():
        matched_loss += (1 - value)

    block_loss /= len(block_f1_keys)
    block_chain_loss /= len(block_chain_f1_keys)
    site_loss /= (len(a_p_abs_diff['mean']) + len(p_a_abs_diff['mean']))
    matched_loss /= len(site_matched['F1'])
    loss = base_loss + block_loss + block_chain_loss + site_loss + matched_loss
    return loss


def get_length_thresholds(path_helper, trained_id, std_num, other_coef):
    if 1 <= other_coef or other_coef < 0:
        raise Exception("Invalid other_coef value")
    length_gaussian_path = path_helper.get_length_log10_model_path(trained_id)
    length_model = pd.read_csv(length_gaussian_path, sep='\t')
    main_kwargs = pd.read_csv(
        path_helper.get_main_kwargs_path()).set_index('name')
    main_kwargs = main_kwargs.to_dict('index')
    upstream_dist = int(main_kwargs['upstream_dist']['value'])
    downstream_dist = int(main_kwargs['downstream_dist']['value'])
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
    length_thresholdss['other'] = min(
        upstream_dist, downstream_dist) * other_coef
    return length_thresholdss


class Reviser:
    def __init__(self, input_raw_plus_gff_path, fasta_path, region_table_path):
        self._input_raw_plus_gff_path = input_raw_plus_gff_path
        self._fasta_path = fasta_path
        self._region_table_path = region_table_path

    def revise(self, saved_root, **kwargs):
        gff_revise_main(saved_root, self._input_raw_plus_gff_path,
                        self._region_table_path, self._fasta_path, **kwargs)


class Evaluator:
    def __init__(self, answer_path, region_table_path):
        self._answer_path = answer_path
        self._region_table_path = region_table_path

    def evaluate(self, predict_path, saved_root):
        loss_path = os.path.join(saved_root, 'overall_loss.txt')
        if not os.path.exists(loss_path):
            performance_main(predict_path, self._answer_path,
                             self._region_table_path,
                             saved_root, chrom_target='new_id')
            overall_loss = get_overall_loss(saved_root)
            with open(loss_path, "w") as fp:
                fp.write("{}\n".format(overall_loss))
        else:
            with open(loss_path, "r") as fp:
                overall_loss = float(fp.read())
        return overall_loss


class ReviseEvaluator:
    def __init__(self, path_helper, trained_id, predicted_root, usage=None):
        self._path_helper = path_helper
        self._region_table_path = path_helper.region_table_path
        self._fasta_path = path_helper.get_fasta_path(trained_id, usage)
        self._raw_plus_gff_path = path = os.path.join(
            predicted_root, 'test_predict_raw_plus.gff3')
        self._answer_path = path_helper.get_answer_path(
            trained_id, usage, on_double_strand=True)
        self._reviser = Reviser(
            self._raw_plus_gff_path,
            self._fasta_path,
            self._region_table_path)
        self._evaluator = Evaluator(self._answer_path, self._region_table_path)

    def process(self, revised_root, **kwargs):
        peformance_root = os.path.join(revised_root, 'test')
        revised_path = os.path.join(revised_root, 'revised.gff3')
        revised_double_strand = os.path.join(
            revised_root, 'revised_double_strand.gff3')
        if not os.path.exists(revised_path):
            self._reviser.revise(revised_root, **kwargs)
        if not os.path.exists(revised_double_strand):
            rename_chrom_main(
                revised_path,
                self._path_helper.region_table_path,
                revised_double_strand)
        return self._evaluator.evaluate(revised_double_strand, peformance_root)


def test(trained_root, test_result_root, path_helper, trained_id, usage=None):
    loss_path = os.path.join(test_result_root, 'test', 'overall_loss.txt')
    if not os.path.exists(loss_path):
        data_path = path_helper.get_processed_data_path(trained_id, usage)
        answer_path = path_helper.get_answer_path(trained_id, usage)
        create_folder(test_result_root)
        data = load_data(data_path)
        best_model, origin_executor = get_best_model_and_origin_executor(
            trained_root)
        batch_size = get_batch_size(trained_root)
        engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
        engine.set_root(test_result_root, with_train=False, with_val=False,
                        create_tensorboard=False)
        engine.test(best_model, origin_executor, data, batch_size=batch_size,
                    region_table_path=path_helper.region_table_path,
                    answer_gff_path=answer_path)
        predict_path = os.path.join(
            test_result_root, 'test', 'test_predict.gff3')
        predict_double_strand = os.path.join(
            test_result_root, 'test', 'test_predict_double_strand.gff3')
        rename_chrom_main(
            predict_path,
            path_helper.region_table_path,
            predict_double_strand)
        overall_loss = get_overall_loss(os.path.join(test_result_root, 'test'))
        with open(loss_path, "w") as fp:
            fp.write("{}\n".format(overall_loss))
    else:
        with open(loss_path, "r") as fp:
            overall_loss = float(fp.read())
    return overall_loss


class AutoReviseEvaluator:
    def __init__(self, path_helper, val_result_root,
                 saved_root, predicted_root):
        self._saved_root = saved_root
        self._val_result_root = val_result_root
        self._path_helper = path_helper
        self._predicted_root = predicted_root
        self._best_hyperparams = None
        self._best_loss = None
        self._records = None
        self._index = None
        self._std_nums = None
        self._other_coefs = None
        self._first_n_motif_list = None
        self._methods_1ist = None
        self._init_revised_space()

    def _init_revised_space(self):
        self._std_nums = [3]
        self._distance_scales = [0,0.3,0.6,0.9]
        self._other_coefs = [0, 0.5]
        self._first_n_motif_list = [4]
        self._methods_1ist = [['length_threshold', 'distance_threshold'],
                              ['distance_threshold', 'length_threshold']]

    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['std_nums'] = self._std_nums
        config['distance_scales'] = self._distance_scales
        config['other_coefs'] = self._other_coefs
        config['first_n_motif_list'] = self._first_n_motif_list
        config['methods_1ist'] = self._methods_1ist
        return config

    def _create_execute_revise_evaluator(
            self, target, methods, length_thresholds, distances, first_n_motif):
        print("Working on {}".format(target))
        revised_root = os.path.join(
            self._saved_root,
            'revised_train_val',
            target,
            'testing')
        revise_evaluator = ReviseEvaluator(self._path_helper,
                                           self._path_helper.train_val_name,
                                           self._val_result_root)
        splicing_kwargs = get_splicing_kwargs(self._path_helper,
                                              first_n=first_n_motif)
        revise_evaluator.process(revised_root, methods=methods,
                                 length_thresholds=length_thresholds,
                                 acceptor_distance=distances['acceptor_distance'],
                                 donor_distance=distances['donor_distance'],
                                 **splicing_kwargs)
        print("Finish working on {}".format(target))

    def _revise_and_evaluate_on_val(self, methods, std_num=None, other_coef=None,
                                    distance_scale=None, first_n_motif=None):
        self._index += 1
        length_thresholds = None
        distances = {'acceptor_distance': None, 'donor_distance': None}
        target = 'revised_train_val_{}'.format(self._index)
        hyperparams = {'std_num': std_num, 'other_coef': other_coef,
                       'distance_scale': distance_scale, 'methods': methods,
                       'first_n_motif': first_n_motif,
                       'target': target}
        self._records.append(hyperparams)
        if distance_scale is not None:
            distances = get_distances(os.path.join(self._val_result_root,'test'),
                                      distance_scale)
        if std_num is not None and other_coef is not None:
            length_thresholds = get_length_thresholds(self._path_helper,
                                                      self._path_helper.train_val_name,
                                                      std_num, other_coef)

        process = Process(target=self._create_execute_revise_evaluator,
                          args=(target, methods, length_thresholds, distances, first_n_motif))
        return process

    def _process_on_test(self, trained_name):
        target = '{}_testing'.format(trained_name)
        print("Working on {}".format(target))
        length_thresholds = None
        distances = {'acceptor_distance': None, 'donor_distance': None}
        methods = self._best_hyperparams['methods']
        std_num = self._best_hyperparams['std_num']
        other_coef = self._best_hyperparams['other_coef']
        distance_scale = self._best_hyperparams['distance_scale']
        first_n_motif = self._best_hyperparams['first_n_motif']
        splicing_kwargs = get_splicing_kwargs(
            self._path_helper, first_n=first_n_motif)
        if distance_scale is not None:
            distances = get_distances(os.path.join(self._val_result_root,'test'),
                                      distance_scale)
        if std_num is not None and other_coef is not None:
            length_thresholds = get_length_thresholds(self._path_helper,
                                                      self._path_helper.train_val_name,
                                                      std_num, other_coef)
        predict_result_root = os.path.join(self._predicted_root, trained_name,
                                           'testing','test')
        revised_root = os.path.join(self._saved_root, 'revised_test',
                                    trained_name,'testing')
        self._test_revise_evaluator = ReviseEvaluator(
            self._path_helper, trained_name, predict_result_root, usage='testing')
        loss = self._test_revise_evaluator.process(revised_root, methods=methods,
                                                   length_thresholds=length_thresholds,
                                                   acceptor_distance=distances['acceptor_distance'],
                                                   donor_distance=distances['donor_distance'],
                                                   **splicing_kwargs)
        print("Finish working on {}".format(target))
        record = dict(self._best_hyperparams)
        record.update({'loss': loss, 'target': target})
        self._records.append(record)

    def process(self):
        self._best_hyperparams = {'std_num': None, 'other_coef': None, 'distance_scale': None,
                                  'methods': None, 'first_n_motif': None}
        self._index = 0
        self._records = []
        overall_loss_path = os.path.join(
            self._val_result_root, 'test', 'overall_loss.txt')
        with open(overall_loss_path, "r") as fp:
            val_loss = float(fp.read())
        val_record = dict(self._best_hyperparams)
        val_record.update({'loss': val_loss, 'target': 'train_val'})
        self._best_loss = val_loss

        processes = []
        for std_num in self._std_nums:
            for other_coef in self._other_coefs:
                for distance_scale in self._distance_scales:
                    for first_n_motif in self._first_n_motif_list:
                        for methods in self._methods_1ist:
                            process = self._revise_and_evaluate_on_val(methods, std_num, other_coef,
                                                                       distance_scale=distance_scale,
                                                                       first_n_motif=first_n_motif)
                            processes.append(process)

        for std_num in self._std_nums:
            for other_coef in self._other_coefs:
                for first_n_motif in self._first_n_motif_list:
                    methods = ['length_threshold']
                    process = self._revise_and_evaluate_on_val(methods, std_num, other_coef,
                                                               first_n_motif=first_n_motif)
                    processes.append(process)

        for distance_scale in self._distance_scales:
            for first_n_motif in self._first_n_motif_list:
                methods = ['distance_threshold']
                process = self._revise_and_evaluate_on_val(methods, distance_scale=distance_scale,
                                                           first_n_motif=first_n_motif)
                processes.append(process)

        print("Run {} proceeses".format(len(processes)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        for record in self._records:
            overall_loss_path = os.path.join(self._saved_root, 'revised_train_val',
                                             record['target'], 'testing', 'test', 'overall_loss.txt')
            with open(overall_loss_path, "r") as fp:
                loss = float(fp.read())
            record['loss'] = loss
            if self._best_loss > loss:
                self._best_hyperparams = record
                self._best_loss = loss

        self._records.append(val_record)
        print("Working on test dataset")
        data_usage = get_data_names(self._path_helper.split_root)
        for trained_name in data_usage.keys():
            self._process_on_test(trained_name)

        record_path = os.path.join(self._saved_root, 'record.tsv')
        pd.DataFrame.from_dict(
            self._records).to_csv(
            record_path,
            sep='\t',
            index=None)
        write_json(
            self._best_hyperparams,
            os.path.join(
                self._saved_root,
                'best_revised_config.json'))


def main(raw_data_root, processed_root, trained_project_root, saved_root):
    path_helper = PathHelper(raw_data_root, processed_root)
    data_usage = get_data_names(path_helper.split_root)
    val_gffs = []
    val_raw_plus_gffs = []
    predicted_root = os.path.join(saved_root, 'predicted')
    merged_val_root = os.path.join(saved_root, 'merged_val')

    for usage in ['validation', 'testing']:
        for trained_name in data_usage.keys():
            trained_root = os.path.join(trained_project_root, trained_name)
            result_root = os.path.join(predicted_root, trained_name, usage)
            torch.cuda.empty_cache()
            test(trained_root, result_root, path_helper, trained_name, usage)
            torch.cuda.empty_cache()

    for trained_name in data_usage.keys():
        result_root = os.path.join(predicted_root, trained_name, 'validation')
        gff = read_gff(os.path.join(result_root, 'test', 'test_predict.gff3'))
        raw_plus_gff = read_gff(os.path.join(result_root,'test',
                                             'test_predict_raw_plus.gff3'))
        val_gffs.append(gff)
        val_raw_plus_gffs.append(raw_plus_gff)

    val_gffs = pd.concat(val_gffs)
    val_raw_plus_gffs = pd.concat(val_raw_plus_gffs)
    create_folder(merged_val_root)
    predicted_path = os.path.join(merged_val_root, 'test_predict.gff3')
    predicted_raw_plus_path = os.path.join(merged_val_root, 
                                           'test_predict_raw_plus.gff3')
    predicted_double_strand_path = os.path.join(
        merged_val_root, 'test_predict_double_strand.gff3')
    write_gff(val_gffs, predicted_path)
    write_gff(val_raw_plus_gffs, predicted_raw_plus_path)

    train_val_answer_path = path_helper.get_answer_path(
        path_helper.train_val_name, on_double_strand=True)
    rename_chrom_main(predicted_path,path_helper.region_table_path,
                      predicted_double_strand_path)
    evaluator = Evaluator(train_val_answer_path, path_helper.region_table_path)
    evaluator.evaluate(predicted_double_strand_path,
                       os.path.join(merged_val_root,'test'))

    processor = AutoReviseEvaluator(path_helper,merged_val_root,
                                    saved_root,predicted_root)
    write_json(processor.get_config(),
               os.path.join(saved_root,'auto_revised_config.json'))
    processor.process()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-r", "--raw_data_root", required=True,
                        help="Root of Arabidopsis processed data")
    parser.add_argument("-p", "--processed_root", required=True,
                        help='The root of processed data by select_data')
    parser.add_argument("-t", "--trained_project_root", required=True,
                        help='The root of trained project')
    parser.add_argument("-s", "--saved_root", required=True,
                        help="The root to save result")
    parser.add_argument("-g", "--gpu_id", type=int,
                        default=0, help="GPU to used")
    args = parser.parse_args()
    kwargs = dict(vars(args))
    del kwargs['gpu_id']
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
