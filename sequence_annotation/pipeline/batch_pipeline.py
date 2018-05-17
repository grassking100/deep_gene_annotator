from abc import ABCMeta, abstractmethod
from . import PipelineFactory
import copy
class BatchPipeline(metaclass=ABCMeta):
    def __init__(self,data_type='simple',is_prompt_visible=True):
        self._train_val_pipeline = None
        self._test_pipeline = None
        self._work_settings = {}
        self._is_prompt_visible = is_prompt_visible
        self._cross_val_number = None
        self.data_type = data_type
        self._id = None
        self._model_setting = None
    def print_prompt(self,value):
        if self._is_prompt_visible:
            print(value)
    def _create_pipeline(self):
        self._train_val_pipeline = PipelineFactory().create('train',self.data_type,
                                                            self._is_prompt_visible)
        self._test_pipeline = PipelineFactory().create('test',self.data_type,
                                                       self._is_prompt_visible)
    def _batch_train_val_execute(self):
        for mode_id in range(1,self._cross_val_number+1):
            work_setting = copy.deepcopy(self._work_settings[str(mode_id)])
            del work_setting["data_path"]["training_validation"]
            del work_setting['data_path']['testing']
            del work_setting["trained_model_path"]
            work_setting["load_model"]=False
            work_setting["validation_split"]=None
            self._train_val_pipeline.execute(self._id,work_setting,self._model_setting)
    def _batch_test_execute(self):
        for mode_id in range(1,self._cross_val_number+1):
            work_setting = copy.deepcopy(self._work_settings[str(mode_id)])
            del work_setting["data_path"]["training_validation"]
            del work_setting["data_path"]["training"]
            del work_setting["data_path"]["validation"]
            work_setting["load_model"]=True
            self._test_pipeline.execute(self._id,work_setting,self._model_setting)
    def _parse_work_setting(self,work_setting):
        training_val_input = work_setting["data_path"]["training_validation"]['inputs']
        training_val_answer = work_setting["data_path"]["training_validation"]['answers']
        testing_input = work_setting["data_path"]["testing"]['inputs']
        testing_answer = work_setting["data_path"]["testing"]['answers']
        self._cross_val_number = len(training_val_input)
        path_root = work_setting['path_root']
        epoch = work_setting['epoch']
        length = str(len(str(epoch)))
        for mode_id in range(1,self._cross_val_number+1):
            temp = copy.deepcopy(work_setting)
            temp['mode_id'] = 'mode_'+str(mode_id)
            inputs = []
            for index in range(1,self._cross_val_number+1):
                if index != mode_id:
                    inputs.append(training_val_input[index-1])
            path = (path_root+"/"+str(self._id)+"/mode_"+str(mode_id)+'/model/epoch_{:0'+length+'d}.h5').format(epoch)
            temp['trained_model_path']=path
            temp['data_path']['training']={}
            temp['data_path']['validation']={}
            temp['data_path']['testing']={}
            temp['data_path']['training']['inputs'] = inputs
            temp['data_path']['training']['answers'] = training_val_answer
            temp['data_path']['validation']['inputs'] = training_val_input[mode_id-1]
            temp['data_path']['validation']['answers'] = training_val_answer
            temp['data_path']['testing']['inputs'] = testing_input
            temp['data_path']['testing']['answers'] = testing_answer
            self._work_settings[str(mode_id)] = temp
    def execute(self,id_,work_setting,model_setting):
        self._id = id_
        self._model_setting = model_setting
        self._parse_work_setting(work_setting)
        self._create_pipeline()
        self._batch_train_val_execute()
        self._batch_test_execute()