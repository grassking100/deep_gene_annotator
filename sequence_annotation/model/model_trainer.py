"""This submodule provides trainer to train model"""
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from . import DataGenerator
from . import ModelWorker
from . import DataValidator, DictValidator, AttrValidator
from . import ResultHistory
class ModelTrainer(ModelWorker):
    """a trainer which will train and evaluate the model"""
    @property
    def valid_data_keys(self):
        return ['train_x','train_y','validation_x','validation_y']
    @property
    def valid_setting_keys(self):
        return ['initial_epoch','batch_size','shuffle','epochs','is_prompt_visible',
                'is_verbose_visible','period','file_path_root','weights']
    def _data_validate(self):
        data_validator = DataValidator()
        data_validator.same_shape(self._data['train_x'],self._data['train_y'])
        data_validator.same_shape(self._data['validation_x'],self._data['validation_y'])
    def _attr_validate(self):
        attr_validator = AttrValidator(self,False,True,False,None)
        attr_validator.validate()
    def _settings_validate(self):
        keys = self.valid_setting_keys
        keys.remove('weights')
        dict_validator = DictValidator(self._settings,
                                       self.valid_setting_keys,keys,[None])
        dict_validator.validate()
    def _validate(self):
        """Validate required data"""
        self._attr_validate()
        self._data_validate()
        self._settings_validate()
    def _get_call_backs(self):
        call_backs = []
        length = str(len(str(self.settings['epochs'])))
        root_path = './'+self.settings['file_path_root']
        call_backs.append(TensorBoard(log_dir=root_path+"/log", histogram_freq=1,
                                      write_graph=True, write_grads=True,
                                      write_images=True))
        call_backs.append(ModelCheckpoint(filepath=root_path+"/model/epoch_{epoch:0"+length+"d}.h5",
                                          verbose=int(self.settings['is_prompt_visible']),
                                          save_best_only=False,
                                          period=self.settings['period']))
        call_backs.append(ResultHistory(filepath=root_path+"/result/epoch_{epoch:0"+length+"d}.csv",
                                        verbose=int(self.settings['is_prompt_visible']),
                                        period=self.settings['period'],previous_results=self.result))
        return call_backs
    def _before_train(self):
        root_path = './'+self.settings['file_path_root']
        length = str(len(str(self.settings['epochs'])))
        for folder_name in ["model","log","result"]:
            path = root_path+"/"+folder_name
            if not os.path.exists(path):
                os.makedirs(path)
        self._model.save((root_path+'/model/epoch_{:0'+length+'d}.h5').format(0))
    def _prepare_data(self,x_data,y_data,batch_size):
        generator = DataGenerator(x_data,y_data,batch_size)
        return generator
    def train(self):
        """Train model"""
        self._validate()
        self._before_train()
        #training and evaluating the model
        train_data = self._prepare_data(self._data['train_x'],
                                        self._data['train_y'],
                                        self.settings['batch_size'])
        val_data = self._prepare_data(self._data['validation_x'],
                                      self._data['validation_y'],
                                      self.settings['batch_size'])
        history = self.model.fit_generator(train_data,
                                           epochs=self.settings['epochs'],
                                           verbose=int(self.settings['is_verbose_visible']),
                                           callbacks=self._get_call_backs(),
                                           validation_data=val_data,
                                           max_queue_size=10,
                                           workers=1,
                                           use_multiprocessing=False,
                                           shuffle=self.settings['shuffle'],
                                           initial_epoch=int(self.settings['initial_epoch']))
        """history = self.model.fit(train_data,
                                 initial_epoch = self.settings['initial_epoch'],
                                 batch_size=self.settings['batch_size'],
                                 shuffle=self.settings['shuffle'],
                                 epochs=self.settings['epochs'],
                                 verbose=int(self.settings['is_verbose_visible']),
                                 validation_data=val_data,
                                 callbacks=self._get_call_backs())"""
        """(self._data['validation_x'],self._data['validation_y'])"""
        self.result = history.history.items()