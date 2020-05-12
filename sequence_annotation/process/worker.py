"""This submodule provides trainer to train model"""
import signal
import warnings
import os
import time
import torch
import numpy as np
from abc import ABCMeta, abstractmethod
from ..utils.utils import create_folder, print_progress, write_json, get_time_str
from .callback import MeanRecorder, Callbacks, DataHolder
from .warning import WorkerProtectedWarning
from .checkpoint import build_checkpoint

GRADIENT_VERBOSE = "Epoch {} of {}: the abs max value is {}, the abs min value is {}, the variance is {}"
GRADIENT_WARNING = "Epoch {} of {}: abs max value is {} larger than one"
L2_VERBOSE = "Epoch {}'s gradient L2 norm is {}"


def _batch_process(model, seq_data, process, callbacks, **kwargs):
    torch.cuda.empty_cache()
    callbacks.on_batch_begin()
    returned = process(model,seq_data,**kwargs)
    torch.cuda.empty_cache()
    with torch.no_grad():
        callbacks.on_batch_end(predicts=returned['predicts'],
                               outputs=returned['outputs'],
                               seq_data=seq_data,
                               metric=returned['metric'],
                               masks=returned['masks'])
    torch.cuda.empty_cache()


def train_per_batch(model, seq_data, executor, callbacks):
    _batch_process(model, seq_data, executor.fit, callbacks)


def evaluate_per_batch(model, seq_data, executor, callbacks):
    _batch_process(model, seq_data, executor.evaluate,
                   callbacks,accumulate=True)

    
def predict_per_batch(model, seq_data, executor, callbacks):
    torch.cuda.empty_cache()
    callbacks.on_batch_begin()
    returned = executor.predict(model,seq_data)
    torch.cuda.empty_cache()
    with torch.no_grad():
        callbacks.on_batch_end(predicts=returned['predicts'],
                               outputs=returned['outputs'],
                               seq_data=seq_data,
                               masks=returned['masks'],
                               metric=None)
    torch.cuda.empty_cache()
    

def validate_gradient(epoch,model,grad_warn_recorder=None,
                      grad_recorder=None):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2.)
        total_norm += param_norm.item()**2.
    total_norm = total_norm**(1. / 2.)
    if grad_warn_recorder is not None:
        l2_verbose = L2_VERBOSE.format(epoch, total_norm)
        grad_warn_recorder.notify([l2_verbose])

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.cpu().detach().numpy()
            if (np.abs(grad) >1).any() and grad_warn_recorder is not None:
                warning = GRADIENT_WARNING.format(epoch, name,
                                                  np.abs(grad).max())
                grad_warn_recorder.notify([warning])
            if grad_recorder is not None:
                verbose = GRADIENT_VERBOSE.format(epoch, name,
                                                  np.abs(grad).max(),
                                                  np.abs(grad).min(),
                                                  grad.var())
                grad_recorder.notify([verbose])


class MessageRecorder:
    def __init__(self, path):
        self.path = path

    def notify(self, messages):
        if not isinstance(messages, list):
            messages = [messages]
        with open(self.path, 'a+') as fp:
            for message in messages:
                fp.write("{}\n".format(message))


class Worker(metaclass=ABCMeta):
    def __init__(self, model, executor, root=None):
        super().__init__()
        self.model = model
        self.executor = executor
        self.result = {}
        self.is_verbose_visible = True
        self._settings = {}
        self.root = root
        if self.root is not None:
            create_folder(self.root)
        self._message_recorder = None
        signal.signal(signal.SIGTERM, self.handle_signal)

    def work(self):
        """Work"""
        self._before_work()
        self._work()
        self._after_work()

    @abstractmethod
    def _work(self):
        """Work"""

    def _after_work(self):
        """Do something after worker work"""

    def _before_work(self):
        """Do something before worker work"""

    def print_verbose(self, info, is_progress=False):
        if self.is_verbose_visible:
            if is_progress:
                print_progress(info)
            else:
                print(info)

    def handle_signal(self, signum, frame):
        pass


class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self,model,executor,train_loader,val_loader,
                 writer=None,epoch=None,root=None,
                 checkpoint_kwargs=None):
        super().__init__(model, executor, root=root)
        self._train_loader = train_loader
        self._val_loader = val_loader
        self.train_callbacks = Callbacks()
        self.val_callbacks = Callbacks()
        self.other_callbacks = Callbacks()
        self.checkpoint_kwargs = checkpoint_kwargs or {}
        self._gradient_recorder = None
        self._gradient_warning_recorder = None
        self._writer = writer
        self._epoch = 1 if epoch is None else epoch
        self._current_epoch = None
        self._best_epoch = None
        self._best_result = None
        self._is_running = True
        self._checkpoint = None

    @property
    def best_result(self):
        return self._best_result

    @property
    def is_running(self):
        return self._is_running

    @is_running.setter
    def is_running(self, value):
        warnings.warn("Is_running SHOULD only be called by callback",
                      WorkerProtectedWarning)
        self._is_running = value

    @property
    def best_epoch(self):
        return self._best_epoch

    def _save_setting(self):
        if self.root is not None:
            for key in [
                    'train_callbacks', 'val_callbacks', 'other_callbacks',
                    'root', '_epoch', 'executor', 'checkpoint_kwargs'
            ]:
                attr = getattr(self, key)
                if hasattr(attr, 'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr
            root = os.path.join(self.root, 'settings')
            create_folder(root)
            path = os.path.join(root, "train_worker_setting.json")
            write_json(self._settings, path)

    def _before_work(self):
        if self.root is not None:
            checkpoint_root = os.path.join(self.root, 'checkpoint')
            self._checkpoint = build_checkpoint(checkpoint_root,
                                                is_train_mode=True,
                                                **self.checkpoint_kwargs)
            self._gradient_warning_recorder = MessageRecorder(
                path=os.path.join(self.root, "gradient_warning.txt"))
            self._message_recorder = MessageRecorder(
                path=os.path.join(self.root, "message.txt"))
            self._gradient_recorder = MessageRecorder(
                path=os.path.join(self.root, "gradient.txt"))
        self.train_callbacks.add(MeanRecorder())
        self.val_callbacks.add(DataHolder(prefix='val'))
        self._is_running = True
        self._save_setting()

    def handle_signal(self, signum, frame):
        self._is_running = False
        warning = "STOP worker at epoch {} by signal {}".format(
            self._current_epoch, signum)
        self.print_verbose(warning)
        if self._message_recorder is not None:
            self._message_recorder.notify([warning])

    def work(self):
        # Execute worker
        pre_time = time.time()
        try:
            super().work()
        except Exception as e:
            if self._message_recorder is not None:
                self._message_recorder.notify([str(e)])
            raise
        time_spend = time.time() - pre_time
        time_messgae = "Time spend: {} seconds".format(time_spend)
        if self._message_recorder is not None:
            self._message_recorder.notify([time_messgae])

    def _work(self):
        """Train model"""
        start = 1
        end = self._epoch + 1
        all_callbacks = Callbacks([
            self.train_callbacks, self.val_callbacks, self.other_callbacks
        ])
        if self._checkpoint is not None:
            all_callbacks.add(self._checkpoint)

        all_callbacks.on_work_begin(worker=self)
        self.executor.on_work_begin()

        if self._checkpoint is not None:
            start = self._checkpoint.epoch_start + 1

        batch_info = "Epoch: ({}/{}), {} {:.1f}% of data"
        epoch_info = "Epoch: ({}/{}), Time cost of: {}, {}\n"
        self.print_verbose("Start from {} to {}".format(start, end - 1))
        if not self.is_running:
            self.print_verbose("Stop at {}".format(start))
        else:
            time_messgae = "Start trainging at {}".format(get_time_str())
            self.print_verbose(time_messgae)
            if self._message_recorder is not None:
                self._message_recorder.notify(time_messgae)
            save_distribution = self.model.save_distribution
            for epoch in range(start, end):
                self._current_epoch = epoch
                pre_time = time.time()
                if self._writer is not None:
                    self._writer.counter = epoch
                all_callbacks.on_epoch_begin(counter=epoch)
                self.model.save_distribution = False
                for index, seq_data in enumerate(self._train_loader):
                    train_per_batch(self.model, seq_data, self.executor,
                                    self.train_callbacks)
                    status = 100 * index / len(self._train_loader)
                    self.print_verbose(batch_info.format(epoch, self._epoch,
                                                         'training',status), True)
                    validate_gradient(epoch, self.model,
                                      self._gradient_warning_recorder,
                                      self._gradient_recorder)

                for index, seq_data in enumerate(self._val_loader):
                    evaluate_per_batch(self.model, seq_data, self.executor,
                                       self.val_callbacks)
                    status = 100 * index / len(self._val_loader)
                    self.print_verbose(batch_info.format(epoch, self._epoch,
                                                         'validating',status),True)
                
                self.model.save_distribution = save_distribution
                train_record = self.train_callbacks.get_data()
                val_record = self.val_callbacks.get_data()
                other_record = self.other_callbacks.get_data()
                if str(train_record['loss']) == 'nan':
                    self._is_running = False
                if 'val_loss' in val_record.keys() and str(
                        val_record['val_loss']) == 'nan':
                    self._is_running = False

                record = {}
                record.update(train_record)
                record.update(val_record)
                record.update(other_record)

                # Executor's on_epoch_end must be called first
                self.executor.on_epoch_end(epoch=epoch, metric=record)
                self.train_callbacks.on_epoch_end(metric=train_record)
                self.val_callbacks.on_epoch_end(metric=val_record)
                self.other_callbacks.on_epoch_end(metric=record)
                if self._checkpoint is not None:
                    self._checkpoint.on_epoch_end(metric=record)

                time_cost = round(time.time() - pre_time, 3)
                self.print_verbose(
                    epoch_info.format(epoch, self._epoch, time_cost, record),
                    True)

                if not self.is_running:
                    self.print_verbose("Stop at {}".format(epoch))
                    break
            time_messgae = "Stop training at {}".format(get_time_str())
            self.print_verbose(time_messgae)
            if self._message_recorder is not None:
                self._message_recorder.notify(time_messgae)

        all_callbacks.on_work_end()

    def _after_work(self):
        if self._checkpoint is not None:
            self.result = self._checkpoint.record
            self._best_result = self._checkpoint.best_result
            self._best_epoch = self._checkpoint.best_epoch


class TestWorker(Worker):
    """a worker which will evaluate the model"""
    def __init__(self,model,executor,data_loader,root=None):
        super().__init__(model, executor, root=root)
        self._loader = data_loader
        self.callbacks = Callbacks()
        self._checkpoint = None

    def _save_setting(self):
        if self.root is not None:
            for key in ['callbacks', 'executor', 'root']:
                attr = getattr(self, key)
                if hasattr(attr, 'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr
            root = os.path.join(self.root, 'settings')
            create_folder(root)
            path = os.path.join(root, "test_worker_setting.json")
            write_json(self._settings, path)

    def _before_work(self):
        if self.root is not None:
            self._checkpoint = build_checkpoint(self.root,
                                                only_recorder=True,
                                                force_reset=True)
        self.callbacks.add(DataHolder(prefix='test'))
        self._save_setting()

    def _work(self):
        """Test model"""
        epoch_start = 1
        callbacks = Callbacks(self.callbacks)
        if self._checkpoint is not None:
            self._checkpoint.on_work_begin(worker=self)
        callbacks.on_work_begin(worker=self)
        self.executor.on_work_begin()
        if self._checkpoint is not None:
            self._checkpoint.on_epoch_begin(counter=epoch_start)
        callbacks.on_epoch_begin(counter=epoch_start)
        batch_info = "Testing data: {0:.1f}%"
        save_distribution = self.model.save_distribution
        self.model.save_distribution = False
        for index, seq_data in enumerate(self._loader):
            evaluate_per_batch(self.model, seq_data, self.executor,
                               self.callbacks)
            status = 100 * index / len(self._loader)
            self.print_verbose(batch_info.format(status), True)

        self.model.save_distribution = save_distribution
        record = callbacks.get_data()
        if self._checkpoint is not None:
            self._checkpoint.on_epoch_end(metric=record)
        self.executor.on_epoch_end(epoch=epoch_start, metric=record)
        callbacks.on_epoch_end(metric=record)
        callbacks.on_work_end()
        if self._checkpoint is not None:
            self._checkpoint.on_work_end()

    def _after_work(self):
        if self._checkpoint is not None:
            if hasattr(self._checkpoint, 'record'):
                self.result = self._checkpoint.record
            else:
                self.result = self._checkpoint.data
                
class PredictWorker(Worker):
    """A worker which will predict"""
    def __init__(self,model,executor,data_loader,root=None):
        super().__init__(model, executor, root=root)
        self._loader = data_loader
        self.callbacks = Callbacks()

    def _save_setting(self):
        if self.root is not None:
            for key in ['callbacks', 'executor', 'root']:
                attr = getattr(self, key)
                if hasattr(attr, 'get_config'):
                    attr = attr.get_config()
                self._settings[key] = attr
            root = os.path.join(self.root, 'settings')
            create_folder(root)
            path = os.path.join(root, "predict_worker_setting.json")
            write_json(self._settings, path)

    def _before_work(self):
        self._save_setting()

    def _work(self):
        """Test model"""
        epoch_start = 1
        callbacks = Callbacks(self.callbacks)
        callbacks.on_work_begin(worker=self)
        self.executor.on_work_begin()
        callbacks.on_epoch_begin(counter=epoch_start)
        batch_info = "Predict data: {0:.1f}%"
        save_distribution = self.model.save_distribution
        self.model.save_distribution = False
        for index, seq_data in enumerate(self._loader):
            predict_per_batch(self.model, seq_data, self.executor,
                              self.callbacks)
            status = 100 * index / len(self._loader)
            self.print_verbose(batch_info.format(status), True)
        
        self.model.save_distribution = save_distribution
        self.executor.on_epoch_end(epoch=epoch_start,metric=None)
        callbacks.on_epoch_end(metric=None)
        callbacks.on_work_end()
