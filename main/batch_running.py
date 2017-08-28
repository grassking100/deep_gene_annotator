import configparser
import argparse
from Exon_intron_finder.Training_helper import traning_validation_data_index_selector
import random,time,importlib,math,sys,numpy as np
from Exon_intron_finder.Exon_intron_finder import Convolution_layers_settings,Exon_intron_finder_factory
from Exon_intron_finder.Exon_intron_finder import tensor_end_with_terminal_binary_accuracy,tensor_end_with_terminal_binary_crossentropy
from Exon_intron_finder.Model_evaluator import Model_evaluator
from DNA_Vector.DNA_Vector import code2vec,codes2vec
from Fasta_handler.Fasta_handler import seqs_index_selector,fastas2arr,seqs2dnn_data
from time import gmtime, strftime
from keras.models import load_model
import Exon_intron_finder
import os, errno
__author__ = 'Ching-Tien Wang'
class Setting_parser():
    def __init__(self):
        self.__config = configparser.ConfigParser()
        self.__model_config = configparser.ConfigParser()
    def set_setting_file(self,setting):
        self.setting=setting
    def get_setting_file(self):
        return(self.setting)
    def get_setting(self):
        self.__config.read(self.get_setting_file())
        settings={'show':self.get_show_settings(),'training':self.get_training_settings(),'model':self.get_model_setting()}
        return settings    
    def str2bool(self,value):
        if value=="True":
            return True
        elif value=="False":
            return False
        else:
            assert False, (str)(value)+" is neithor True of False"
    def get_model_setting_file(self):
        config=self.__config
        model_setting_file=config["model_setting"]['model_setting_file']
        return model_setting_file
    def get_model_setting(self):
        model_setting_file=self.get_model_setting_file()
        config=self.__model_config
        config.read(model_setting_file)
        root="model_setting"
        total_convolution_layer_size=config[root]['total_convolution_layer_size']
        LSTM_layer_number=config[root]['LSTM_layer_number']
        add_terminal_signal=self.str2bool(config[root]['add_terminal_signal'])
        convolution_layer_sizes=[int(i) for i in config[root]['convolution_layer_sizes'].split(",")]
        convolution_layer_numbers=[int(i) for i in config[root]['convolution_layer_numbers'].split(",")]
        add_batch_normalize=self.str2bool(config[root]['add_batch_normalize'])
        settings={'total_convolution_layer_size':int(total_convolution_layer_size),'LSTM_layer_number':int(LSTM_layer_number),
                 'add_terminal_signal':add_terminal_signal,'convolution_layer_sizes':convolution_layer_sizes,
                'convolution_layer_numbers':convolution_layer_numbers,
                 'add_batch_normalize':add_batch_normalize}
        return settings
    def get_show_settings(self):
        config=self.__config
        show_settings="show"
        model=self.str2bool(config[show_settings]['model'])
        verbose=self.str2bool(config[show_settings]['verbose'])
        prompt=self.str2bool(config[show_settings]['prompt'])
        settings= {'model':model,'verbose':verbose,'prompt':prompt}
        return settings
    def get_training_settings(self):
        config=self.__config
        training_settings="training_settings"
        training_files=config[training_settings]['training_files'].split(",\n")
        validation_files=config[training_settings]['validation_files'].split(",\n")
        mode_id=config[training_settings]['mode_id']
        train_id=config[training_settings]['train_id']
        cross_validation=config[training_settings]['cross_validation']
        epoch=config[training_settings]['epoch']
        previous_status_root=config[training_settings]['previous_status_root']
        step=config[training_settings]['step']
        batch_size=config[training_settings]['batch_size']
        outputfile_root=config[training_settings]['outputfile_root']
        previous_epoch=config[training_settings]['previous_epoch']
        settings= {'training_files':training_files,
                   'validation_files':validation_files,
                   'mode_id':mode_id,'train_id':train_id,'epoch':int(epoch),
                   'step':int(step),'outputfile_root':outputfile_root,
                   'previous_epoch':int(previous_epoch),
                   'cross_validation':int(cross_validation),
                   'batch_size':int(batch_size),
                   'previous_status_root':previous_status_root}
        return settings
class Batch_ruuning():
    #initialize all the variable needed
    def __init__(self,setting):
        self.__setting=setting
        #initialize all the variable from setting file
        self.__init_parameter()
        #get names and sequences from file
        self.__init_training_files_data()
        self.__init_validation_files_data()
        if self.is_prompt_visible():
            self.print_file_classification()
        #create training and validation set
        self.__init_data()
        if self.is_prompt_visible():
            self.print_selected_information()
        #create model and evaluator
        self.__init_model()
        if self.is_model_visible():
            self.print_model_information()
        self.__init_evaluator()
        self.__restored_previous_result()
        self.__create_stored_folder()
    def __create_stored_folder(self):
        try:
            folder_name=self.__root+'/train_'+str(self.__train_id)
            if not os.path.exists(folder_name):
                print("Create folder:"+folder_name)
                os.makedirs(folder_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    def print_model_information(self):
        print("Create model")
        self.get_model().summary()
    def __restored_previous_result(self):
        if self.__previous_epoch>0:
            if self.is_prompt_visible():
                print("Loading previous status:"+self.__previous_status_root)
            self.__load_previous_result()
        else:
            if self.is_prompt_visible():
                print("Start a new training status")
            return
    def __load_previous_result(self):
        whole_file_path=self.__previous_status_root
        self.get_evaluator().add_previous_histories(np.load(whole_file_path+'.npy').tolist())
        self.__best_model=load_model(whole_file_path+'.h5',custom_objects=
                                     {'tensor_end_with_terminal_binary_accuracy':tensor_end_with_terminal_binary_accuracy,
                                      'tensor_end_with_terminal_binary_crossentropy':tensor_end_with_terminal_binary_crossentropy})
        self.get_evaluator().set_model(self.__best_model)
    def print_file_classification(self):
        print("Status of file")
        for file in self.get_training_files():
            print("    Training set:"+file)
        for file in self.get_validation_files():
            print("    Validation set:"+file)    
    def get_training_files(self):
        return self.__training_files
    def get_validation_files(self):
        return self.__validation_files
    def get_training_seqs(self):
        return self.__training_seqs
    def get_validation_seqs(self):
        return self.__validation_seqs
    def __init_training_files_data(self):
        self.__training_seqs=[]
        #print(self.__training_files)
        for file in self.__training_files:
            (name,seq)=fastas2arr(file)
            self.__training_seqs+=seq
    def __init_validation_files_data(self):
        self.__validation_seqs=[]
        for file in self.__validation_files:
            (name,seq)=fastas2arr(file)
            self.__validation_seqs+=seq
    def __init_training_parameter(self):
        root="training"
        self.__cross_validation=self.__settings[root]['cross_validation']
        self.__mode_id=self.__settings[root]['mode_id']
        self.__training_files=self.__settings[root]['training_files']
        self.__validation_files=self.__settings[root]['validation_files']
        self.__progress_target=self.__settings[root]['epoch']
        self.__previous_epoch=self.__settings[root]['previous_epoch']
        self.__step=self.__settings[root]['step']
        self.__batch_size=self.__settings[root]['batch_size']
        self.__root=self.__settings[root]['outputfile_root']
        self.__train_id=self.__settings[root]['train_id']
        self.__train_file='/train_'+str(self.__train_id)+'/'
        self.__date=strftime("%Y_%b_%d", gmtime())
        self.__previous_status_root=self.__settings[root]['previous_status_root']
    def __init_show_parameter(self):
        root="show"
        self.__is_model_visible=self.__settings[root]['model']
        self.__is_verbose_visible=self.__settings[root]['verbose']
        self.__is_prompt_visible=self.__settings[root]['prompt']
    def __init_model_paramter(self):
        root="model"
        self.__total_convolution_layer_size=self.__settings[root]['total_convolution_layer_size']
        self.__LSTM_layer_number=self.__settings[root]['LSTM_layer_number']
        self.__add_terminal_signal=self.__settings[root]['add_terminal_signal']
        self.__convolution_layer_sizes=self.__settings[root]['convolution_layer_sizes']
        self.__convolution_layer_numbers=self.__settings[root]['convolution_layer_numbers']
        self.__add_batch_normalize=self.__settings[root]['add_batch_normalize']
    def __init_parameter(self):
        parser=Setting_parser()
        parser.set_setting_file(self.__setting)
        self.__settings=parser.get_setting()
        self.__init_training_parameter()
        self.__init_show_parameter()
        self.__init_model_paramter()
    def is_prompt_visible(self):
        return self.__is_prompt_visible
    def is_model_visible(self):  
        return self.__is_model_visible
    def is_verbose_visible(self):
        return self.__is_verbose_visible
    def print_training_status(self):
        print("Training satus:")
        print("    Progress target:"+str(self.__progress_target))
        print("    Start progress:"+str(self.__previous_epoch))
        print("    Step:"+str(self.__step))
        print("    Batch size:"+str(self.__batch_size))
        print("    Train id:"+str(self.__train_id))
        print("    Date:"+str(self.__date))
    def __init_model(self):
        convolution_layers_settings=Convolution_layers_settings()
        for i in range(self.__total_convolution_layer_size):
            convolution_layers_settings.add_layer(self.__convolution_layer_numbers[i],
                                                  self.__convolution_layer_sizes[i])
        model=Exon_intron_finder_factory(convolution_layers_settings.get_settings(),
                                   self.__LSTM_layer_number,
                                   self.__add_terminal_signal,
                                   self.__add_batch_normalize)    
        self.__best_model=model
    def get_model(self):
        return self.__best_model
    def __init_data(self):
        (self.__x_train,self.__y_train)=seqs2dnn_data(self.get_training_seqs(),False)
        (self.__x_validation,self.__y_validation)=seqs2dnn_data(self.get_validation_seqs(),False)
    def get_training_set(self):
        return self.__x_train,self.__y_train
    def get_validation_set(self):
        return self.__x_validation,self.__y_validation
    def __init_evaluator(self):
        self.evaluator=Model_evaluator()
        self.evaluator.set_training_data(*self.get_training_set())
        self.evaluator.set_validation_data(*self.get_validation_set())
        self.evaluator.set_model(self.get_model())
    def get_evaluator(self):
        return self.evaluator
    def print_selected_information(self):
        training_size=len(self.get_training_seqs())
        validation_size=len(self.get_validation_seqs())
        valid_training_size=len(self.__y_train)
        valid_validation_size=len(self.__y_validation)
        print('Selected set number:'+(str)(training_size+validation_size))
        print('Training set number:'+(str)(training_size))
        print('Validation set number:'+(str)(validation_size))
        print('Parsing and validate data...')
        print('Selected valid set number:'+(str)(valid_training_size+valid_validation_size))
        print('Training valid set number:'+(str)(valid_training_size))
        print('Validation valid set number:'+(str)(valid_validation_size))
    def get_while_path_file(self,progress_number):
        file_name='train_'+str(self.__train_id)+'_mode_'+str(self.__mode_id)+'_progress_'+str(progress_number)+'_'
        whole_file_path=self.__root+self.__train_file+file_name+self.__date
        return whole_file_path
    def run(self):   
        if self.is_prompt_visible():
            print("Start of running")
        if self.__previous_epoch==0:
            saved_new_model=self.get_while_path_file(0)
            self.get_model().save(saved_new_model+'.h5')
        for progress in range(self.__previous_epoch,self.__progress_target,self.__step):
            whole_file_path=self.get_while_path_file(self.__step+progress)
            if self.is_prompt_visible():
                print("Starting training:"+whole_file_path)
            self.get_evaluator().evaluate(self.__step,self.__batch_size,True,int(self.is_model_visible()))
            np.save(whole_file_path+'.npy', self.get_evaluator().get_histories()) 
            self.get_model().save(whole_file_path+'.h5')
            if self.is_prompt_visible():
                print("Saved training:"+whole_file_path)
        if self.is_prompt_visible():
            print("End of running")
if __name__=='__main__':
    prompt='batch_running.py --setting=<setting_file>'
    parser = argparse.ArgumentParser(description=prompt)
    parser.add_argument('-s','--setting',help='Setting file name', required=True)
    args = parser.parse_args()
    batch_runner=Batch_ruuning(args.setting)
    batch_runner.run()
    print("End of program")