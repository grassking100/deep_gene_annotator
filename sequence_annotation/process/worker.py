"""This submodule provides trainer to train model"""
import signal
import time
from abc import ABCMeta, abstractmethod
from ..utils.utils import get_time_str, print_progress
from .callback import Callbacks

class Worker(metaclass=ABCMeta):
    def __init__(self, executor):
        super().__init__()
        self._model = executor.model
        self._executor = executor
        self._settings = {}
        self._message_recorder = None
        self._result = {}
        self.is_verbose_visible = True
        signal.signal(signal.SIGTERM, self.handle_signal)

    @property
    def model(self):
        return self._model
        
    @property
    def executor(self):
        return self._executor
        
    @property
    def result(self):
        return self._result
        
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
    def __init__(self,train_executor,val_executor,epoch=None,
                 other_callbacks=None,message_recorder=None):
        super().__init__(train_executor)
        self._train_executor = train_executor
        self._val_executor = val_executor
        self._current_epoch = None
        self._is_running = True
        self._train_callbacks = train_executor.callbacks
        self._val_callbacks = val_executor.callbacks
        self._other_callbacks = other_callbacks or Callbacks()
        self._epoch = epoch or 1
        self._start_epoch = 1

    def set_start_epoch(self,value):
        if value <= 0:
            raise
        self._start_epoch = value

    def set_running_stauts(self, value):
        self._is_running = value

    def _before_work(self):
        self._is_running = True

    def handle_signal(self, signum, frame):
        self._is_running = False
        warning = "STOP worker at epoch {} by signal {}".format(
            self._current_epoch, signum)
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
        all_callbacks = Callbacks([self._train_callbacks, self._val_callbacks, self._other_callbacks])
        all_callbacks.on_work_begin(worker=self)
        start = self._start_epoch
        end = self._epoch
        epoch_info = "Epoch: ({}/{}), Time cost of: {}, {}\n"
        self.print_verbose("Start from {} to {}".format(start, end))
        if not self._is_running:
            self.print_verbose("Stop at {}".format(start))
        else:
            time_messgae = "Start training at {}".format(get_time_str())
            self.print_verbose(time_messgae)
            if self._message_recorder is not None:
                self._message_recorder.notify(time_messgae)
            save_distribution = self._model.save_distribution
            for epoch_counter in range(start, end+1):
                self._current_epoch = epoch_counter
                pre_time = time.time()
                all_callbacks.on_epoch_begin(counter=epoch_counter)
                self._model.save_distribution = False
                self._train_executor.execute()
                self._val_executor.execute()
                self._model.save_distribution = save_distribution
                record = all_callbacks.get_data()
                if str(record['loss']) == 'nan':
                    self._is_running = False
                if 'val_loss' in record.keys() and str(record['val_loss']) == 'nan':
                    self._is_running = False
                self._train_callbacks.on_epoch_end(metric=self._train_callbacks.get_data())
                self._val_callbacks.on_epoch_end(metric=self._val_callbacks.get_data())
                self._other_callbacks.on_epoch_end(metric=record)

                time_cost = round(time.time() - pre_time, 3)
                self.print_verbose(epoch_info.format(epoch_counter, self._epoch, 
                                                     time_cost, record),True)

                if not self._is_running:
                    self.print_verbose("Stop at {}".format(epoch_counter))
                    break
            time_messgae = "Stop training at {}".format(get_time_str())
            self.print_verbose(time_messgae)
            if self._message_recorder is not None:
                self._message_recorder.notify(time_messgae)

        all_callbacks.on_work_end()
        

class BasicWorker(Worker):
    """a worker which will evaluate the model"""
    def __init__(self,executor):
        super().__init__(executor)
        self._callbacks = executor.callbacks

    def _work(self):
        """Test model"""
        self._callbacks.on_work_begin(worker=self)
        self._callbacks.on_epoch_begin(counter=1)
        save_distribution = self._model.save_distribution
        self._model.save_distribution = False
        self._executor.execute()
        self._model.save_distribution = save_distribution
        record = self._callbacks.get_data()
        self._callbacks.on_epoch_end(metric=record)
        self._callbacks.on_work_end()
