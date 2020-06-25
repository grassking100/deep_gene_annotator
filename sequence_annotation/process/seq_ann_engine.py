import os
import abc
import numpy as np
import torch
from ..utils.utils import create_folder, write_json, BASIC_GENE_ANN_TYPES, get_time_str
from ..utils.utils import get_file_name, read_json
from ..utils.seq_converter import SeqConverter
from ..genome_handler.ann_seq_processor import seq2vecs
from ..genome_handler.seq_container import AnnSeqContainer
from .data_processor import AnnSeqProcessor
from .utils import param_num
from .worker import TrainWorker, TestWorker,PredictWorker
from .tensorboard_writer import TensorboardWriter
from .callback import CategoricalMetric, TensorboardCallback, LearningRateHolder
from .callback import SeqFigCallback, Callbacks, ContagionMatrix
from .signal_handler import SignalHandlerBuilder
from .data_generator import SeqGenerator, SeqCollateWrapper
from .convert_signal_to_gff import build_ann_vec_gff_converter
from .model import get_model
from .executor import get_executor


class SeqAnnEngine(metaclass=abc.ABCMeta):
    def __init__(self,channel_order):
        self._settings = {}
        self._root = None
        self._writer = None
        self._train_writer = self._val_writer = self._test_writer = None
        self.is_verbose_visible = True
        self._channel_order = channel_order
        self._ann_types = self._channel_order
        self.batch_size = 1

    def update_settings(self, key, params):
        if isinstance(params, dict):
            params = dict(params)
            if 'self' in params.keys():
                del params['self']
        self._settings[key] = params

    def print_verbose(self, info, *args):
        if self.is_verbose_visible:
            print(info, *args)

    @property
    def ann_types(self):
        if self._ann_types is not None:
            return self._ann_types
        else:
            raise Exception(
                "Object has not set train_ann_seqs or test_ann_seqs yet.")

    def get_seq_fig(self, seq, ann_seq, color_settings, prefix=None):
        if self._writer is None:
            raise Exception("Writer must be set first")
        seq = SeqConverter().seq2vecs(seq)
        seq = np.transpose(np.array([seq]), [0, 2, 1])
        seq = torch.from_numpy(seq).type('torch.FloatTensor').cuda()
        ann_seq = [seq2vecs(ann_seq, self.ann_types)]
        colors = [color_settings[type_] for type_ in self.ann_types]
        seq_fig = SeqFigCallback(self._writer,
                                 seq,
                                 ann_seq,
                                 prefix=prefix,
                                 label_names=self.ann_types,
                                 colors=colors)
        return seq_fig

    def set_root(self,root,with_train=True,with_val=True,
                 with_test=True,create_tensorboard=True,
                 with_predict=False):
        self.update_settings('set_root', locals())
        self._root = root
        create_folder(self._root)
        if create_tensorboard:
            self._writer = TensorboardWriter(self._root)
        if with_train:
            folder = os.path.join(self._root, "train")
            create_folder(folder)
            if create_tensorboard:
                self._train_writer = TensorboardWriter(folder)
        if with_val:
            folder = os.path.join(self._root, "val")
            create_folder(folder)
            if create_tensorboard:
                self._val_writer = TensorboardWriter(folder)
        if with_test:
            folder = os.path.join(self._root, "test")
            create_folder(folder)
            if create_tensorboard:
                self._test_writer = TensorboardWriter(folder)
        if with_predict:
            folder = os.path.join(self._root, "predict")
            create_folder(folder)

    def _record_ann_count(self, name, ann_vecs):
        count = {}
        for type_ in self._channel_order:
            count[type_] = 0
        for vec in ann_vecs:
            for index,item_count in enumerate(vec.sum(0)):
                count[self._channel_order[index]] += item_count
        self.print_verbose(len(ann_vecs), count)
        for key, value in count.items():
            if int(value) == 0:
                raise Exception(
                    "The {} is missing in the dataset".format(key))
        self.update_settings(name, count)

    def get_signal_handler(self,root,prefix=None,inference=None,
                           region_table_path=None,
                           answer_gff_path=None):

        builder = SignalHandlerBuilder(root, prefix=prefix)
        if region_table_path is not None:
            ann_vec_gff_converter = self.create_ann_vec_gff_converter()
            builder.add_converter_args(
                inference,
                region_table_path,
                ann_vec_gff_converter)
        if answer_gff_path is not None:
            builder.add_answer_path(answer_gff_path)
        signal_handler = builder.build()
        return signal_handler

    def _create_categorical_metric(self, prefix=None):
        metric = CategoricalMetric(len(self.ann_types),
                                   label_names=self.ann_types,
                                   prefix=prefix)
        return metric

    def _create_contagion_matrix(self, prefix=None):
        metric = ContagionMatrix(len(self.ann_types),
                                 label_names=self.ann_types,
                                 prefix=prefix)
        return metric

    def _create_default_train_callbacks(self):
        train_callbacks = Callbacks()
        train_metric = self._create_categorical_metric()
        train_matrix = self._create_contagion_matrix()
        learning_rate_holder = LearningRateHolder()
        train_callbacks.add([train_metric, train_matrix, learning_rate_holder])
        val_callbacks = Callbacks()
        val_metric = self._create_categorical_metric(prefix='val')
        val_matrix = self._create_contagion_matrix(prefix='val')
        val_callbacks.add([val_metric, val_matrix])
        return train_callbacks, val_callbacks

    def _create_default_test_callbacks(self):
        callbacks = Callbacks()
        test_metric = self._create_categorical_metric(prefix='test')
        test_matrix = self._create_contagion_matrix(prefix='test')
        callbacks.add([test_metric, test_matrix])
        return callbacks

    def _update_common_setting(self):
        self.update_settings('setting', {
            'ann_types': self._ann_types,
            'channel_order': self._channel_order,
            'batch_size': self.batch_size
        })

    def _update_ann_seqs_count(self, name, origin_count, filtered_count):
        self.update_settings(name, {
            'origin count': origin_count,
            'filtered count': filtered_count
        })

    def _add_tensorboard_callback(self, writer, callbacks, add_grad=False):
        if writer is not None and callbacks is not None:
            tensorboard = TensorboardCallback(writer)
            tensorboard.do_add_grad = add_grad
            callbacks.add(tensorboard)

    def process_data(self, raw_data):
        #simplify_map = simplify_map or BASIC_SIMPLIFY_MAP
        #raw_data = dict(raw_data)
        #for type_,dict_ in raw_data.items():
        #    if 'answers' in dict_:
        #        dict_['answers'] = simplify_genome(dict_['answers'], simplify_map)
        data = AnnSeqProcessor(self._channel_order).process(raw_data)
        keys = list(data.keys())
        for key in keys:
            if 'answers' in data[key]:
                self._update_ann_seqs_count(key, len(raw_data[key]['answers'].ids),
                                            len(data[key]['ids']))
                self._record_ann_count('{}_ann_counut'.format(key),
                                       data[key]['answers'])
                has_gene_statuses_count = int(sum(data[key]['has_gene_statuses']))
                self.update_settings('{}_has_gene_count'.format(key),
                                     has_gene_statuses_count)
        return data

    def create_data_gen(self,collate_fn=None,shuffle=True,*args, **kwargs):
        train_gen = SeqGenerator(batch_size=self.batch_size,
                                 shuffle=shuffle,
                                 seq_collate_fn=collate_fn,*args, **kwargs)
        return train_gen

    def create_basic_data_gen(self):
        test_gen = SeqGenerator(batch_size=self.batch_size,
                                shuffle=False)
        return test_gen

    def create_ann_vec_gff_converter(self, simply_map=None):
        converter = build_ann_vec_gff_converter(self._channel_order,
                                                simply_map=simply_map)
        return converter

    def train(self,model,executor,train_loader,val_loader,
              epoch=None,other_callbacks=None,
              add_grad=True,checkpoint_kwargs=None):
        self._update_common_setting()
        other_callbacks = other_callbacks or Callbacks()
        epoch = epoch or 100
        self.update_settings(
            'train_setting', {
                'epoch': epoch,
                'add_grad': add_grad,
                'checkpoint_kwargs': checkpoint_kwargs,
            })
        # Set callbacks and writer
        train_callbacks, val_callbacks = self._create_default_train_callbacks()
        self._add_tensorboard_callback(self._train_writer,
                                       train_callbacks,
                                       add_grad=add_grad)
        self._add_tensorboard_callback(self._val_writer,
                                       val_callbacks,
                                       add_grad=add_grad)

        # Create worker
        worker = TrainWorker(model,executor,train_loader,val_loader,
                             writer=self._writer,epoch=epoch,root=self._root,
                             checkpoint_kwargs=checkpoint_kwargs)
        worker.train_callbacks = train_callbacks
        worker.val_callbacks = val_callbacks
        worker.other_callbacks = other_callbacks
        worker.is_verbose_visible = self.is_verbose_visible
        # Save setting
        if self._root is not None:
            root = os.path.join(self._root, 'settings')
            create_folder(root)
            param_num_path = os.path.join(root, 'model_param_num.txt')
            if not os.path.exists(param_num_path):
                with open(param_num_path, "w") as fp:
                    fp.write("Required-gradient parameters number:{}".format(
                        param_num(model)))

            setting_path = os.path.join(root, "train_setting.json")
            model_config_path = os.path.join(root, "model_config.json")
            model_component_path = os.path.join(root, "model_component.txt")
            exec_config_path = os.path.join(root, "executor_config.json")
            #if not os.path.exists(setting_path):
            write_json(self._settings, setting_path)
            if not os.path.exists(model_config_path):
                write_json(model.get_config(), model_config_path)
            if not os.path.exists(exec_config_path):
                write_json(executor.get_config(), exec_config_path)
            if not os.path.exists(model_component_path):
                with open(model_component_path, "w") as fp:
                    fp.write(str(model))
        # Execute worker
        worker.work()
        return worker

    def test(self,model,executor,data_loader,callbacks=None):
        self._update_common_setting()
        callbacks = callbacks or Callbacks()
        test_callbacks = self._create_default_test_callbacks()
        callbacks.add(test_callbacks)
        self._add_tensorboard_callback(self._test_writer, callbacks)
        root = os.path.join(self._root, 'test')
        worker = TestWorker(model,executor,data_loader,root=self._root)
        worker.callbacks = callbacks
        if self._root is not None:
            root = os.path.join(self._root, 'settings')
            create_folder(root)
            path = os.path.join(root, "test_setting.json")
            if not os.path.exists(path):
                write_json(self._settings, path)

        worker.work()
        return worker
    
    def predict(self,model,executor,data_loader,callbacks=None):
        self._update_common_setting()
        root = os.path.join(self._root, 'predict')
        worker = PredictWorker(model,executor,data_loader,root=self._root)
        worker.callbacks = callbacks
        if self._root is not None:
            root = os.path.join(self._root, 'settings')
            create_folder(root)
            path = os.path.join(root, "predict_setting.json")
            if not os.path.exists(path):
                write_json(self._settings, path)

        worker.work()
        return worker

    
def _get_first_large_data(data, batch_size):
    seqs, ann_container = data
    lengths = {key: len(seq) for key, seq in seqs.items()}
    print("Max length is {}".format(max(lengths.values())))
    sorted_length_keys = sorted(lengths, key=lengths.get, reverse=True)
    part_keys = sorted_length_keys[:batch_size]
    part_seqs = dict(zip(part_keys, [seqs[key] for key in part_keys]))
    part_container = AnnSeqContainer()
    part_container.ANN_TYPES = ann_container.ANN_TYPES
    for key in part_keys:
        part_container.add(ann_container.get(key))

    return part_seqs, part_container


def check_max_memory_usgae(saved_root, model, executor, train_data, val_data,
                           batch_size,concat=False):
    create_folder(saved_root)
    train_data = _get_first_large_data(train_data, batch_size)
    val_data = _get_first_large_data(val_data, batch_size)
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.is_verbose_visible=False
    engine.batch_size=batch_size
    
    train_seqs, train_ann_seqs = train_data
    val_seqs, val_ann_seqs = val_data
    raw_data = {
        'training': {'inputs': train_seqs,'answers': train_ann_seqs},
        'validation':{'inputs': val_seqs, 'answers': val_ann_seqs}
    }
    data = engine.process_data(raw_data)
    train_gen = engine.create_data_gen(SeqCollateWrapper(concat=concat))
    val_gen = engine.create_basic_data_gen()
    train_loader = train_gen(data['training'])
    val_loader = val_gen(data['validation'])
    try:
        torch.cuda.reset_max_memory_cached()
        engine.train(model,executor,train_loader,val_loader,epoch=1)
        max_memory = torch.cuda.max_memory_reserved()
        messenge = "Max memory allocated is {}\n".format(max_memory)
        print(messenge)
        path = os.path.join(saved_root, 'max_memory.txt')
        with open(path, "w") as fp:
            fp.write(messenge)
    except RuntimeError:
        path = os.path.join(saved_root, 'error.txt')
        with open(path, "a") as fp:
            fp.write("Memory might be fulled at {}\n".format(get_time_str()))
        raise Exception("Memory is fulled")


def get_model_executor(model_config_path,executor_config_or_path,
                       model_weights_path=None,
                       executor_weights_path=None,
                       frozen_names=None,
                       save_distribution=False):
    # Create model
    model = get_model(model_config_path,
                      model_weights_path=model_weights_path,
                      frozen_names=frozen_names,
                      save_distribution=save_distribution)
    # Create executor
    executor = get_executor(model,executor_config_or_path,
                            executor_weights_path=executor_weights_path)
    return model, executor


def get_best_model_and_origin_executor(saved_root,latest=False):
    setting = read_json(os.path.join(saved_root, 'main_setting.json'))
    executor_config_path = os.path.join(
        saved_root, 'resource',
        get_file_name(setting['executor_config_path'], True))
    model_config_path = os.path.join(
        saved_root, 'resource',
        get_file_name(setting['model_config_path'], True))
    if latest:
        model_weights_path = os.path.join(saved_root, 'checkpoint',
                                          'latest_model.pth')
    else:
        model_weights_path = os.path.join(saved_root, 'checkpoint',
                                          'best_model.pth')
    model, executor = get_model_executor(model_config_path,
                                         executor_config_path,
                                         model_weights_path=model_weights_path)
    return model, executor


def get_batch_size(saved_root):
    setting = read_json(os.path.join(saved_root, 'main_setting.json'))
    batch_size = setting['batch_size']
    return batch_size
