"""This submodule provides trainer to train model"""
import warnings
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
from keras.utils import plot_model
from ...process.worker import Worker
from ...process.data_generator import DataGenerator
from ..function.work_generator import FitGenerator


class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self,train_generator=None,val_generator=None,fit_generator=None):
        super().__init__()
        self._fit_generator = fit_generator
        self._train_generator = train_generator
        self._val_generator = val_generator
        if self._fit_generator is None:
            self._fit_generator = FitGenerator()
        if self._train_generator is None:
            self._train_generator = DataGenerator()
        if self._val_generator is None:
            self._val_generator = DataGenerator()
    def before_work(self,path=None):
        if path is not None:
            create_folder(path+"/model")
            create_folder(path+"/log")
            create_folder(path+"/result")
            plot_saved_path = path+"/model_image.png"
            plot_model(self.model,show_shapes=True,to_file=plot_saved_path)
            length = str(len(str(self._epoch)))
            model_saved_path = path+'/model/epoch_{:0'+length+'d}.h5'
            model_saved_path = model_saved_path.format(0)
            self.model.save(model_saved_path)

    def work(self):
        """Train model"""
        self._validate()
        self._train_generator.x_data= self.data['training']['inputs']
        self._train_generator.y_data=self.data['training']['answers']
        if 'validation' in self.data.keys():
            self._val_generator.x_data= self.data['validation']['inputs']
            self._val_generator.y_data=self.data['validation']['answers']
        self._fit_generator.train_generator = self._train_generator
        self._fit_generator.val_generator = self._val_generator
        self._fit_generator.model = self.model
        result =  self._fit_generator()
        self.result = result.history

    def after_work(self,path=None):
        warnings.warn("This method is deprecated,it will be replaced by using callback in Keras",
                      DeprecationWarning)
        if path is not None:
            data = json.loads(pd.Series(self._result).to_json(orient='index'))
            with open(path + '/result/record.json', 'w') as outfile:  
                json.dump(data, outfile,indent=4)
