from keras.callbacks import Callback
import pandas as pd
from . import NotPositiveException
class ResultHistory(Callback):
    def __init__(self,filepath,verbose,period,previous_results):
        super().__init__()
        self._filepath = filepath
        self._verbose = verbose
        self._previous_results = previous_results
        if period <= 0:
            raise NotPositiveException("period",period)
        self._period = period
        self._count= None
        self._results = None
    def on_train_begin(self, logs={}):
        self._results = self._previous_results or {}
    def _add_data(self, logs):
        for key,value in logs.items():
            if key not in self._results.keys():
                self._results[key] = []
            else:
                self._results[key].append(value)
    def _save_data(self, epoch):
        log_dir = self._filepath.format(epoch=epoch)
        df = pd.DataFrame().from_dict(self._results)
        df.to_csv(log_dir,index=False)
    def on_epoch_end(self, epoch, logs={}):
        self._add_data(logs)
        if (epoch+1)%self._period==0:
            if self._verbose:
                print("Save result to "+self._filepath.format(epoch=epoch+1))
            self._save_data(epoch+1)