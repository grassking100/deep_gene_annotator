"""This submodule provides class to defined test pipeline"""
import pandas as pd
import os
import json
from time import gmtime, strftime
from os.path import expanduser
from abc import abstractmethod,abstractproperty
from . import SeqAnnDataHandler
from . import SimpleDataHandler
from . import Pipeline
from . import TestWorker

class BasicPipeline(Pipeline):
    @abstractmethod
    def _get_class_count(self):
        pass
    @abstractproperty
    def _setting_saved_name(self):
        pass 
    def _prepare_for_compile(self):
        weight_setting = self._work_setting['class_weight_setting']
        if weight_setting['use_weight']:
            is_static = weight_setting['is_static']
            if is_static:
                class_counts = class_counts = self._get_class_count()
                self._class_weight=self.data_handler.get_weight(class_counts=class_counts,
                                                                method_name=weight_setting['method'])
                    
            else:
                self._dynamic_weight_method = weight_setting['method']
    
    def _save_setting(self):
        data =  self._setting_to_saved()
        mode_id=self._work_setting['mode_id']
        data['save_time'] = strftime("%Y_%b_%d", gmtime())
        path_root=self._work_setting['path_root'] + "/" + str(self._id) + "/" + mode_id
        if not os.path.exists(path_root):
            os.makedirs(path_root)
        with open(path_root + self._setting_saved_name, 'w') as outfile:
            json.dump(data, outfile,indent=4)
