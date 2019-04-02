from keras.callbacks import Callback,ModelCheckpoint
from keras.utils import plot_model
import pandas as pd
from ..utils.exception import NotPositiveException
import pandas as pd

class ResultHistory(Callback):
    def __init__(self,file_path, period, verbose=True, previous_results=None):
        super().__init__()
        self._file_path = file_path
        self._verbose = verbose
        self._previous_results = previous_results or {}
        if period <= 0:
            raise NotPositiveException("period",period)
        self._period = period
        self._done_epoch= None
        self._results = self._previous_results
        self._last_logs = None
    def on_train_end(self, logs=None):
        """Check if data have been save or not.If it haven't then save it"""
        if (self._done_epoch % self._period) != 0:
            if self._verbose:
                print("Save last result to " + self._file_path.format(epoch=self._done_epoch))
            self._add_data(self._last_logs)
            self._results['epoch'].append(self._done_epoch)
            self._save_data()
    def _add_data(self, logs):
        for key,value in logs.items():
            if key not in self._results.keys():
                self._results[key] = []           
            self._results[key].append(value)
    def _save_data(self):
        log_dir = self._file_path.format(epoch=self._done_epoch)
        df = pd.DataFrame().from_dict(self._results)
        df.to_csv(log_dir,index=False)
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._last_logs = logs
        self._done_epoch = epoch + 1
        if (self._done_epoch % self._period) == 0:
            self._add_data(logs)
            if 'epoch' not in self._results.keys():
                self._results['epoch'] = []
            self._results['epoch'].append(self._done_epoch)
            if self._verbose:
                print("Save result to "+self._file_path.format(epoch=self._done_epoch))
            self._save_data()
            
class AdvancedModelCheckpoint(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch,logs)
        self._done_epoch = epoch + 1
    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        """Check if data have been save or not.If it haven't then save it"""
        if self.verbose:
            print("Save first model to " + self.filepath.format(epoch=0))
        self.model.save(self.filepath.format(epoch=0))
    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        """Check if data have been save or not.If it haven't then save it"""
        if self._done_epoch % self.period != 0 and not self.save_best_only:
            if self.verbose:
                print("Save last model to " + self._file_path.format(epoch=self._done_epoch))
            if self.save_weights_only:
                model.save_weights(self.filepath.format(epoch=self._done_epoch))
            else:
                model.save(self.filepath.format(epoch=self._done_epoch))

class ModelPlot(Callback):
    def __init__(self,file_path, verbose=True,*args,**kwargs):
        super().__init__()
        self._file_path = file_path
        self._verbose = verbose
        self._args=args
        self._kwargs=kwargs
    def on_train_begin(self, logs=None):
        if self._verbose:
            print('Save model picture to '+self._file_path)
        plot_model(self.model,to_file=self._file_path,*self._args,**self._kwargs)