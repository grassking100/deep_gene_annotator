from . import Exon_intron_finder_factory
from . import Model_evaluator

class Exon_intron_finder_optimizer:
    def __init__(self,convolution_settings,LSTM_layer_num_range):
        self.convolution_settings=convolution_settings
        self.LSTM_layer_num_range=LSTM_layer_num_range
        self.reset_records()
    def __init_setting(self):
        for convolution_setting in self.convolution_settings:
            for LSTM_layer_num in self.LSTM_layer_num_range:
                setting={'convolution_setting':convolution_setting,'LSTM_layer_num':LSTM_layer_num}
                self.settings.append(setting)
    def reset_records(self):
        self.settings=[]
        self.accuracies=[]
        self.val_accuracies=[]
    #input model and related setting,return largest accuracy
    def eval(self,epoches,x_training,y_training,x_testing,y_testing,batch_size,shuffle,verbose,show_message=True):
        self.reset_records()
        self.__init_setting()
        for setting in self.settings:
            if show_message:
                print('Running model:'+(str)(setting)+'...')
            model=Exon_intron_finder_factory(
                setting['convolution_setting'],
                setting['LSTM_layer_num'])
            evaluator=Model_evaluator()
            evaluator.add_traning_data(x_training,y_training)
            evaluator.add_validation_data(x_testing,y_testing)
            evaluator.evaluate(model,epoches,batch_size,shuffle,verbose)
            accuracy=evaluator.get_last_accuracy()
            val_accuracy=evaluator.get_last_validation_accuracy()
            self.accuracies.append(accuracy)
            self.val_accuracies.append(val_accuracy)
            if show_message:
                print('    last accuracy:'+(str)(accuracy))
                print('    last val_accuracy:'+(str)(val_accuracy))
    def get_model(self):
        setting=self.get_setting()
        if setting is not None:
            convolution_setting=setting['convolution_setting']
            LSTM_layer_num=setting['LSTM_layer_num']
            model=Exon_intron_finder_factory(
                convolution_setting,
                LSTM_layer_num
            )
            return model
        return None
    def get_setting(self,based_on_training=True):
        optimized_index=self.get_index(based_on_training)
        if optimized_index==-1:
            return None
        return self.settings[optimized_index]
    def get_index(self,based_on_training=True):
        optimized_index=-1
        if based_on_training:
            if len(self.accuracies)>0:
                optimized_index=self.accuracies.index(max(self.accuracies))
        else:
            if len(self.val_accuracies)>0:
                optimized_index=self.val_accuracies.index(max(self.val_accuracies))
        return optimized_index
    def get_accuracy(self,based_on_training=True):
        optimized_index=self.get_index(based_on_training)
        if optimized_index!=-1:
            if based_on_training:
                return self.accuracies[optimized_index]
            else:
                return self.val_accuracies[optimized_index]                           
        return None
    def get_accuracies(self,based_on_training=True):
            if based_on_training:
                return self.accuracies
            else:
                return self.val_accuracies