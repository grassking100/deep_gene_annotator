"""This submodule provides trainer to train model"""
from keras import backend as K
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from . import DataGenerator
from . import ModelWorker
from . import ResultHistory
class ModelTrainer(ModelWorker):
    """a trainer which will train and evaluate the model"""
    def __init__(self):
        self.data = None
        self.model = None
        self._period = None
        self._epoch = None
        self._batch_size = None
        self._shuffle = None
        self._initial_epoch = None
        self.callbacks = []
        super().__init__()
    def init_worker(self,path_root, epoch,
                    batch_size,shuffle=True,
                    initial_epoch=0,period=1,
                    validation_split=0.0,
                    previous_result=None,
                    is_verbose_visible=True,
                    is_prompt_visible=True):
        super().init_worker(path_root,is_verbose_visible=is_verbose_visible,
                            is_prompt_visible=is_prompt_visible)
        self._period = period
        self._epoch = epoch
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._initial_epoch = initial_epoch
        self._validation_split = validation_split
        self.result = previous_result
        
    def _validate(self):
        """Validate required data"""
        pass
    def _get_addition_callbacks(self):
        callbacks = []
        length = str(len(str(self._epoch)))
        root_path = './'+self._path_root
        """callbacks.append(TensorBoard(log_dir=root_path+"/log", histogram_freq=0,
                                     write_graph=True, write_grads=True,
                                     write_images=True))"""
        callbacks.append(ModelCheckpoint(filepath=root_path+"/model/epoch_{epoch:0"+length+"d}.h5",
                                         verbose=int(self._is_prompt_visible),
                                         save_best_only=False,
                                         period=self._period,
                                         save_weights_only=False))
        callbacks.append(ResultHistory(filepath=root_path+"/result/epoch_{epoch:0"+length+"d}.csv",
                                       verbose=int(self._is_prompt_visible),
                                       period=self._period,previous_results=self.result))
        return callbacks
    def before_work(self):
        super().before_work()
        root_path = './'+self._path_root
        self._create_folder(["model","log","result"])
        plot_model(self.model, show_shapes=True,to_file=root_path+"/model_image.png")
        length = str(len(str(self._epoch)))
        self.model.save((root_path+'/model/epoch_{:0'+length+'d}.h5').format(0))
    def _prepare_data(self,x_data,y_data,batch_size):
        generator = DataGenerator(x_data,y_data,batch_size)
        return generator
    def _train_by_generator(self):
        train_x = self.data['training']['inputs']
        train_y = self.data['training']['answers']
        train_data = self._prepare_data(train_x,train_y,self._batch_size)
        if 'validation' in self.data.keys():
            val_x = self.data['validation']['inputs']
            val_y = self.data['validation']['answers']
            val_data = self._prepare_data(val_x,val_y,self._batch_size)
        else:
            val_data = None
        callbacks = self._get_addition_callbacks()+self.callbacks
        history = self.model.fit_generator(train_data,
                                           epochs=self._epoch,
                                           verbose=int(self._is_verbose_visible),
                                           callbacks=callbacks,
                                           validation_data=val_data,
                                           shuffle=self._shuffle,
                                           initial_epoch=self._initial_epoch)
        return history
    def _train_by_fit(self):
        train_x = self.data['training']['inputs']
        train_y = self.data['training']['answers']
        if 'validation' in self.data.keys():
            val_x = self.data['validation']['inputs']
            val_y = self.data['validation']['answers']
            val = (val_x,val_y)
        else:
            val = None
        callbacks = self._get_addition_callbacks()+self.callbacks            
        history = self.model.fit(x=train_x,y=train_y,
                                 epochs=self._epoch,
                                 verbose=int(self._is_verbose_visible),
                                 callbacks=callbacks,
                                 validation_data=val,
                                 batch_size=self._batch_size,
                                 shuffle=self._shuffle,
                                 initial_epoch=self._initial_epoch)
        return history
    def after_work(self):
        pass
    def work(self):
        """Train model"""
        self._validate()
        #training and evaluating the model
        history = self._train_by_generator()
        self.result = history.history.items()