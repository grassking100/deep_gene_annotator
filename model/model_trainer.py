from . import keras
from . import numpy
#a trainer which will train and evaluate the model
class ModelTrainer:
    def __init__(self):
        self.histories={}
        self.padding_signal=None
    def set_training_data(self,x,y):
        self.__x_train=x
        self.__y_train=y
        return self
    def set_validation_data(self,x,y):
        self.__x_validation=x
        self.__y_validation=y
        return self
    def clean_histories(self):
        self.histories={}
    @property
    def model(self):
        return self.__model
    @model.setter
    def model(self,v):
        self.__model=v
    @property
    def histories(self):
        return self.__histories
    @histories.setter
    def histories(self,v):
        self.__histories=v
    @property
    def padding_signal(self):
        return self.__padding_signal
    @padding_signal.setter
    def padding_signal(self,v):
        self.__padding_signal=v
    def train(self,epoches,batch_size,shuffle,verbose,log_file):
        if self.padding_signal is not None:
        #padding data to same length
            x_train=keras.preprocessing.sequence.pad_sequences(self.__x_train, maxlen=None,padding='post')
            y_train=keras.preprocessing.sequence.pad_sequences(self.__y_train, maxlen=None,padding='post',value=self.padding_signal)
            x_validation=keras.preprocessing.sequence.pad_sequences(self.__x_validation, maxlen=None,padding='post')
            y_validation=keras.preprocessing.sequence.pad_sequences(self.__y_validation, maxlen=None,padding='post',value=self.padding_signal)
        else:
            x_train=self.__x_train
            y_train=self.__y_train
            x_validation=self.__x_validation
            y_validation=self.__y_validation
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./'+log_file, histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
        tbCallBack.set_model(self.model)
        #training and evaluating the model
        history=self.model.fit(numpy.array(x_train), 
                                     numpy.array(y_train), 
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     epochs=epoches,
                                     verbose=verbose,
                                     validation_data=(numpy.array(x_validation),numpy.array(y_validation)),
                                     callbacks=[tbCallBack])
        #add record to histories
        for k,v in history.history.items():
            if k in self.histories.keys():
                self.histories[k]+=v
            else:
                self.histories[k]=v
        return self