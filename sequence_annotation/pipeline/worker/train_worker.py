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
from keras.utils import plot_model
from .worker import Worker
class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self,path_root=None,is_verbose_visible=True):
        super().__init__(path_root)
        self._result = {}
        self.is_verbose_visible = is_verbose_visible
        self.wrapper = None

    def before_work(self):
        if self._path_root is not None:
            root_path = './'+self._path_root
            create_folder(root_path+"/model")
            create_folder(root_path+"/log")
            create_folder(root_path+"/result")
            plot_saved_path = root_path+"/model_image.png"
            plot_model(self.model,show_shapes=True,to_file=plot_saved_path)
            length = str(len(str(self._epoch)))
            model_saved_path = root_path+'/model/epoch_{:0'+length+'d}.h5'
            model_saved_path = model_saved_path.format(0)
            self.model.save(model_saved_path)

    def work(self):
        """Train model"""
        self._validate()
        history = self.wrapper(self.model,self.data)
        self._result = history.history
