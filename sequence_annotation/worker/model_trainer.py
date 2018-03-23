"""This submodule provides trainer to train model"""
from keras import backend as K
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
                    previous_result=None,
                    is_verbose_visible=True,
                    is_prompt_visible=True):
        super().init_worker(path_root,is_verbose_visible=is_verbose_visible,
                            is_prompt_visible=is_prompt_visible)
        self._period = int(period)
        self._epoch = int(epoch)
        self._batch_size =int(batch_size)
        self._shuffle =bool(shuffle)
        self._initial_epoch = initial_epoch
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
        train_data = self._prepare_data(self.data['training']['inputs'],
                                        self.data['training']['answers'],
                                        self._batch_size)
        val_data = self._prepare_data(self.data['validation']['inputs'],
                                      self.data['validation']['answers'],
                                      self._batch_size)
        return self.model.fit_generator(train_data,
                                        epochs=self._epoch,
                                        verbose=int(self._is_verbose_visible),
                                        callbacks=self._get_addition_callbacks()+self.callbacks,
                                        validation_data=val_data,
                                        max_queue_size=10,
                                        workers=1,
                                        use_multiprocessing=False,
                                        shuffle=self._shuffle,
                                        initial_epoch=self._initial_epoch)
    def _train_by_fit(self):
        return self.model.fit(x=self.data['training']['inputs'],
                              y=self.data['training']['answers'],
                              epochs=self._epoch,
                              verbose=int(self._is_verbose_visible),
                              callbacks=self._get_addition_callbacks()+self.callbacks,
                              validation_data=(self.data['validation']['inputs'],
                                               self.data['validation']['answers']),
                              batch_size=self._batch_size,
                              shuffle=self._shuffle,
                              initial_epoch=self._initial_epoch)
    def after_work(self):
        pass
    def work(self):
        """Train model"""
        self._validate()
        #training and evaluating the model
        history = self._train_by_generator()
        self.result = history.history.items()