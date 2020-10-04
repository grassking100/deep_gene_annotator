"""This submodule provides trainer to train model"""
import time
import signal
from abc import ABCMeta, abstractmethod
from ..utils.utils import get_time_str, print_progress


class Worker(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self._settings = {}
        self._message_recorder = None
        self._result = {}
        self.is_verbose_visible = True
        self._is_running = True
        signal.signal(signal.SIGTERM, self.handle_signal)

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
        self._is_running = True

    def print_verbose(self, info, is_progress=False):
        if self.is_verbose_visible:
            if is_progress:
                print_progress(info)
            else:
                print(info)

    def handle_signal(self, signum, frame):
        pass

    def set_running_stauts(self, value):
        self._is_running = value
        
    def set_start_epoch(self, value):
        pass

class TrainWorker(Worker):
    """
        A worker which will train and evaluate the model
        #The other_callbacks on_batch_begin and on_batch_end won't be called by worker
    """
    def __init__(self,train_executor,val_executor,
                 other_executor=None,epoch=None,
                 message_recorder=None):
        super().__init__()
        self._train_executor = train_executor
        self._val_executor = val_executor
        self._other_executor = other_executor
        self._current_epoch = None
        self._epoch = epoch or 1
        self._start_epoch = 1


    def set_start_epoch(self,value):
        if value <= 0:
            raise
        self._start_epoch = value
        

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
        #Call on_work_begin
        self._train_executor.on_work_begin(worker=self)
        self._val_executor.on_work_begin(worker=self)
        self._other_executor.on_work_begin(worker=self)
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
            for epoch_counter in range(start, end+1):
                self._current_epoch = epoch_counter
                pre_time = time.time()
                #Call on_epoch_begin
                self._train_executor.on_epoch_begin(counter=epoch_counter)
                self._val_executor.on_epoch_begin(counter=epoch_counter)
                self._other_executor.on_epoch_begin(counter=epoch_counter)
                #Execute
                self._train_executor.execute()
                self._val_executor.execute()
                self._other_executor.execute()
                #Get record
                train_record = self._train_executor.callbacks.get_data()
                val_record = self._val_executor.callbacks.get_data()
                record = dict(train_record)
                record.update(val_record)
                if str(record['loss']) == 'nan':
                    self._is_running = False
                if 'val_loss' in record.keys() and str(record['val_loss']) == 'nan':
                    self._is_running = False
                #Call on_epoch_end
                self._train_executor.on_epoch_end(metric=train_record)
                self._val_executor.on_epoch_end(metric=val_record)
                self._other_executor.on_epoch_end(metric=record)
                time_cost = round(time.time() - pre_time, 3)
                self.print_verbose(epoch_info.format(epoch_counter, self._epoch, time_cost, record),True)
                if not self._is_running:
                    self.print_verbose("Stop at {}".format(epoch_counter))
                    break
                    
            time_messgae = "Stop training at {}".format(get_time_str())
            self.print_verbose(time_messgae)
            if self._message_recorder is not None:
                self._message_recorder.notify(time_messgae)

        #Call on_work_end
        self._train_executor.on_work_end()
        self._val_executor.on_work_end()
        self._other_executor.on_work_end()
        

class BasicWorker(Worker):
    """a worker which will evaluate the model"""
    def __init__(self,executor):
        super().__init__()
        self._executor = executor

    def _work(self):
        """Test model"""
        self._executor.on_work_begin(worker=self)
        self._executor.on_epoch_begin(counter=1)
        self._executor.execute()
        record = self._executor.callbacks.get_data()
        self._executor.on_epoch_end(metric=record)
        self._executor.on_work_end()
