from . import keras
from . import numpy
#a trainer which will train and evaluate the model
class ModelTrainer:
    def __init__(self):
        self.x_train=[]
        self.y_train=[]
        self.x_validation=[]
        self.y_validation=[]
        self.histories=[]
    def set_training_data(self,x,y):
        self.x_train=x
        self.y_train=y
        return self
    def set_validation_data(self,x,y):
        self.x_validation=x
        self.y_validation=y
        return self
    def clean_histories(self):
        self.histories=[]
    def add_previous_histories(self,histories):
        self.histories=histories
    def set_model(self,model):
        self.model=model
        return self
    def get_model(self):
        return self.model
    def evaluate(self,epoches,batch_size,shuffle,verbose,log_file):
        #padding data to same length
        
        x_train=keras.preprocessing.sequence.pad_sequences(self.x_train, maxlen=None,padding='post')
        print(numpy.array(x_train).shape)
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
        #add record to histories
        self.histories.append(history.history)
        return self
    def get_histories(self):
        return self.histories
    def get_accuracies(self):
        return self.get_histories()['tensor_end_with_terminal_binary_accuracy']
    def get_validation_accuracies(self):
        return self.get_histories()['val_tensor_end_with_terminal_binary_accuracy']
    def get_losses(self):
        return self.get_histories()['loss']
    def get_validation_losses(self):
        return self.get_histories()['val_loss']
    def get_last_validation_loss(self):
        loss=self.get_validation_losses()
        return loss[len(loss)-1]
    def get_last_loss(self):
        loss=self.get_losses()
        return loss[len(loss)-1]
    def get_last_accuracy(self):
        acc=self.get_accuracies()
        return acc[len(acc)-1]
    def get_last_validation_accuracy(self):
        acc=self.get_validation_accuracies()
        return acc[len(acc)-1]   
    def get_max_accuracy(self):
        return max(self.get_accuracies())
    def get_max_validation_accuracy(self):
        return max(self.get_validation_accuracies())
    

    