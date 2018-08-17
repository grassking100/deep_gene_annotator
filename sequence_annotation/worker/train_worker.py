"""This submodule provides trainer to train model"""
from keras import backend as K
import numpy as np
import tensorflow as tf
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
from keras.utils import Sequence
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from . import DataGenerator
from . import Worker
from . import ResultHistory
class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self, path_root, epoch,
                 batch_size,model,data,
                 previous_result=None,
                 shuffle=True,
                 initial_epoch=0,
                 period=1,
                 validation_split=0.0,
                 use_generator=False):
        super().__init__(path_root)
        self._model = model
        self._data = data
        self._result = previous_result or {}
        self._epoch = epoch
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._initial_epoch = initial_epoch
        self._validation_split = validation_split
        self._use_generator = use_generator
        self._period = period
        self.callbacks = []

    def _validate(self):
        """Validate required data"""
        pass
        
    def _get_addition_callbacks(self):
        callbacks = []
        length = str(len(str(self._epoch)))
        root_path = './'+self._path_root
        
        if self._val_x is None and self._val_y is None:
            val_data = None
        elif self._val_x is not None and self._val_y is not None:
            if self._use_generator:
                val_data = self._prepare_data(self._val_x,self._val_y,self._batch_size)
            else:
                val_data = (self._val_x,self._val_y)
        else:
            raise Exception("Data is missing")
        tensor_board = TensorBoard(log_dir=root_path+"/log",
                                   batch_size=self._batch_size,
                                   histogram_freq=0,
                                   write_graph=True,
                                   write_grads=False,write_images=True)
        callbacks.append(tensor_board)
        model_saved_path = root_path+"/model/epoch_{epoch:0"+length+"d}.h5"
        callbacks.append(ModelCheckpoint(filepath=model_saved_path,
                                         verbose=int(self.is_prompt_visible),
                                         save_best_only=False,
                                         period=self._period,
                                         save_weights_only=False))
        result_saved_path = root_path+"/result/epoch_{epoch:0"+length+"d}.csv"
        callbacks.append(ResultHistory(filepath=result_saved_path,
                                       verbose=int(self.is_prompt_visible),
                                       period=self._period,
                                       previous_results=self._result))
        return callbacks
        
    def before_work(self):
        root_path = './'+self._path_root
        self._create_folder(["model","log","result"])
        plot_saved_path = root_path+"/model_image.png"
        plot_model(self._model,
                   show_shapes=True,
                   to_file=plot_saved_path)
        length = str(len(str(self._epoch)))
        model_saved_path = root_path+'/model/epoch_{:0'+length+'d}.h5'
        model_saved_path = model_saved_path.format(0)
        self._model.save(model_saved_path)
        self._process_data()
    def _prepare_data(self,x_data,y_data,batch_size):
        generator = DataGenerator(x_data,y_data,batch_size)
        return generator
    def _train_by_generator(self):
        if self._val_x is None and self._val_y is None:
            val_data = None
        elif self._val_x is not None and self._val_y is not None :
            val_data = self._prepare_data(self._val_x,self._val_y,self._batch_size)
        else:
            raise Exception("Data is missing")
        val_data = self._prepare_data(self._val_x,self._val_y,self._batch_size)
        train_data = self._prepare_data(self._train_x,self._train_y,self._batch_size)
        callbacks = self._get_addition_callbacks()+self.callbacks
        history = self._model.fit_generator(train_data,
                                            epochs=self._epoch,
                                            workers = 3,
                                            use_multiprocessing=True,
                                            verbose=int(self.is_verbose_visible),
                                            callbacks=callbacks,
                                            validation_data=val_data,
                                            shuffle=self._shuffle,
                                            initial_epoch=self._initial_epoch)
        return history
    def _process_data(self):
        if self._use_generator:
            if 'validation' in self._data.keys():
                train_x = self._data['training']['inputs']
                train_y = self._data['training']['answers']
                val_x = self._data['validation']['inputs']
                val_y = self._data['validation']['answers']
            else:
                data_x = self._data['training']['inputs']
                data_y = self._data['training']['answers']
                shuffled_index = list(range(len(data_x)))
                np.random.shuffle(shuffled_index)
                index = int(len(data_x)*(1-self._validation_split))
                train_index = shuffled_index[:index]
                val_index = shuffled_index[index:]
                val_x = [data_x[i] for i in val_index]
                val_y = [data_y[i] for i in val_index]
                train_x = [data_x[i] for i in train_index]
                train_y = [data_y[i] for i in train_index]
        else:
            train_x = self._data['training']['inputs']
            train_y = self._data['training']['answers']
            if 'validation' in self._data.keys():
                val_x = self._data['validation']['inputs']
                val_y = self._data['validation']['answers']
                val = (val_x,val_y)
            else:
                val = None
        self._train_x = train_x
        self._train_y = train_y
        self._val_x = val_x
        self._val_y = val_y
    def _train_by_fit(self):
        if self._val_x is None and self._val_y is None:
            val = None
        elif self._val_x is not None and self._val_y is not None :
            val = (self._val_x,self._val_y)
        else:
            raise Exception("Data is missing")
        callbacks = self._get_addition_callbacks()+self.callbacks
        history = self._model.fit(x=self._train_x,y=self._train_y,
                                  epochs=self._epoch,
                                  verbose=int(self.is_verbose_visible),
                                  callbacks=callbacks,
                                  validation_data=val,
                                  batch_size=self._batch_size,
                                  shuffle=self._shuffle,
                                  validation_split=self._validation_split,
                                  initial_epoch=self._initial_epoch)
        return history
        
    def after_work(self):
        pass
        
    def work(self):
        """Train model"""
        self._validate()
        if self._use_generator:
            history = self._train_by_generator()
        else:
            history = self._train_by_fit()
        self._result = history.history.items()
