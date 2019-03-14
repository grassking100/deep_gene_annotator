"""This submodule provides trainer to train model"""
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
if hasattr(config,"gpu_options"):
    config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)
from keras.utils import plot_model
from ...process.worker import Worker
from ...function.data_generator import DataGenerator


class TrainWorker(Worker):
    """a worker which will train and evaluate the model"""
    def __init__(self,path_root=None,*args,**kwargs):
        super().__init__(path_root)
        self._args = args
        self._kwargs = kwargs
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
        data = {}
        for data_kind,item in self.data.items():
            if 'batch_size' in self._kwargs:
                batch_size = self._kwargs['batch_size']
            else:
                batch_size = 32
            data[data_kind] = DataGenerator(item['inputs'],item['answers'],batch_size=batch_size)
        if 'validation' not in data.keys():
            data['validation'] = None
        if 'batch_size' in self._kwargs:
            del self._kwargs['batch_size']
        result =  self.model.fit_generator(generator=data['training'],
                                      validation_data=data['validation'],
                                      *self._args,**self._kwargs)
        self._result = result.history
    def after_work(self):
        if self._path_root is not None:
            data = json.loads(pd.Series(self._result).to_json(orient='index'))
            with open(self._path_root + '/result/record.json', 'w') as outfile:  
                json.dump(data, outfile,indent=4)
