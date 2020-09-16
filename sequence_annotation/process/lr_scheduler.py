from torch.optim.lr_scheduler import ReduceLROnPlateau
from .callback import Callback,get_prefix,DataCallback


class LRScheduler:
    def __init__(self,optimizer,patience=None,factor=None,target=None,warmup_epoch=None):
        self._optimizer = optimizer
        self._warmup_epoch = warmup_epoch or 0
        self._threshold = 0
        self._patience = patience or 10 
        self._factor = factor or 0.5
        self._target = 'val_loss'
        self._lr_history = {}
        self._lr_scheduler = ReduceLROnPlateau(optimizer,verbose=True,
                                               threshold=self._threshold,
                                               factor=self._factor,
                                               patience=self._patience - 1)
        
    def step(self, counter, metrics):
        for index, group in enumerate(self._optimizer.param_groups):
            if index not in self._lr_history:
                self._lr_history[index] = []
            self._lr_history[index].append(group['lr'])
        if counter > self._warmup_epoch:
            self._lr_scheduler.step(metrics[self._target])
        
    def get_config(self):
        config = {}
        config['class_name'] = self.__class__.__name__
        config['lr_scheduler_name'] = self._lr_scheduler.__class__.__name__
        config['lr_scheduler'] = self._lr_scheduler.state_dict()
        config['target'] = self._target
        config['patience'] = self._patience
        return config

    def state_dict(self):
        state_dict = {}
        state_dict['lr_history'] = self._lr_history
        state_dict['lr_scheduler'] = self._lr_scheduler.state_dict()
        return state_dict
      
    def load_state_dict(self, state_dicts):
        self._lr_history = state_dicts['lr_history']
        self._lr_scheduler.load_state_dict(state_dicts['lr_scheduler'])

class LRSchedulerCallback(Callback):
    def __init__(self,lr_scheduler):
        self._scheduler = lr_scheduler
        self._counter = None
            
    def get_config(self):
        config = super().get_config()
        config.update(self._scheduler.get_config())
        return config
            
    def on_epoch_begin(self, counter):
        self._counter = counter
        
    def on_epoch_end(self, metric):
        self._scheduler.step(self._counter,metric)

class LearningRateHolder(DataCallback):
    def __init__(self, prefix=None):
        self._prefix = get_prefix(prefix)
        self._executor = None

    def on_work_begin(self, worker, **kwargs):
        self._executor = worker.executor

    def get_config(self):
        config = super().get_config()
        config['prefix'] = self._prefix
        return config

    def get_data(self):
        data = {}
        for index, group in enumerate(self._executor.optimizer.param_groups):
            data["{}lr_{}".format(self._prefix, index)] = group['lr']
        return data
