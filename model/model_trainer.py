from . import keras
from . import numpy
#a trainer which will train and evaluate the model
class ModelTrainer:
    def __init__(self):
        self.histories={}
    def set_training_data(self,x,y):
        self.x_train=x
        self.y_train=y
        return self
    def set_validation_data(self,x,y):
        self.x_validation=x
        self.y_validation=y
        return self
    def clean_histories(self):
        self.histories={}
    def add_previous_histories(self,histories):
        self.histories=histories
    def set_model(self,model):
        self.model=model
        return self
    def get_model(self):
        return self.model
    def train(self,epoches,batch_size,shuffle,verbose,log_file):
        #padding data to same length
        x_train=keras.preprocessing.sequence.pad_sequences(self.x_train, maxlen=None,padding='post')
        y_train=keras.preprocessing.sequence.pad_sequences(self.y_train, maxlen=None,padding='post',value=-1)
        x_validation=keras.preprocessing.sequence.pad_sequences(self.x_validation, maxlen=None,padding='post')
        y_validation=keras.preprocessing.sequence.pad_sequences(self.y_validation, maxlen=None,padding='post',value=-1)
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./'+log_file, histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
        tbCallBack.set_model(self.model)
        #training and evaluating the model
        history=self.get_model().fit(numpy.array(x_train), 
                                     numpy.array(y_train), 
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     epochs=epoches,
                                     verbose=verbose,
                                     validation_data=(numpy.array(x_validation),numpy.array(y_validation)),
                                     callbacks=[tbCallBack])
        #print(history.history)
        #
        #add record to histories
        for k,v in history.history.items():
            #print(k)
            #print(v)
            if k in self.histories.keys():
                self.histories[k]+=v
            else:
                self.histories[k]=v
        #print(self.histories)
        return self
    def get_histories(self):
        return self.histories
    def get_accuracies(self):
        return self.get_histories()['accuracy']
    def get_validation_accuracies(self):
        return self.get_histories()['val_accuracy']
    def get_losses(self):
        return self.get_histories()['loss']
    def get_validation_losses(self):
        return self.get_histories()['val_loss']
    def get_last_validation_loss(self):
        warnings.warn("this function is deprecated,it will be removed in the future",
        PendingDeprecationWarning)
        loss=self.get_validation_losses()
        return loss[len(loss)-1]
    def get_last_loss(self):
        warnings.warn("this function is deprecated,it will be removed in the future",
        PendingDeprecationWarning)
        loss=self.get_losses()
        return loss[len(loss)-1]
    def get_last_accuracy(self):
        warnings.warn("this function is deprecated,it will be removed in the future",
        PendingDeprecationWarning)
        acc=self.get_accuracies()
        return acc[len(acc)-1]
    def get_last_validation_accuracy(self):
        warnings.warn("this function is deprecated,it will be removed in the future",
        PendingDeprecationWarning)
        acc=self.get_validation_accuracies()
        return acc[len(acc)-1]   
    def get_max_accuracy(self):
        warnings.warn("this function is deprecated,it will be removed in the future",
        PendingDeprecationWarning)
        return max(self.get_accuracies())
    def get_max_validation_accuracy(self):
        warnings.warn("this function is deprecated,it will be removed in the future",
        PendingDeprecationWarning)
        return max(self.get_validation_accuracies())
    

    