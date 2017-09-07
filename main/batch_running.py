import configparser
import argparse
import os, errno
import sys
#sys.path.append("~/../")
from Exon_intron_finder.Training_helper import traning_validation_data_index_selector
import random,time,importlib,math,sys,numpy as np
from Exon_intron_finder.Exon_intron_finder import Convolution_layers_settings,Exon_intron_finder_factory
from Exon_intron_finder.Exon_intron_finder import tensor_end_with_terminal_binary_crossentropy
from Exon_intron_finder.Exon_intron_finder import tensor_end_with_terminal_binary_accuracy
from Exon_intron_finder.Model_evaluator import Model_evaluator
from DNA_Vector.DNA_Vector import code2vec,codes2vec
from Fasta_handler.Fasta_handler import seqs_index_selector,fastas2arr,seqs2dnn_data
from time import gmtime,strftime
from keras.models import load_model
from keras.utils import plot_model
import Exon_intron_finder
import csv
__author__ = 'Ching-Tien Wang'
def str2bool(value):
    if value=="True":
        return True
    elif value=="False":
        return False
    else:
        assert False, (str)(value)+" is neithor True of False"
class Setting_parser(object):
    def __init__(self):
        self._config=configparser.ConfigParser()
    @property
    def setting_file(self):
        return self.__setting_file
    @property
    def config(self):
        return self._config    
    @setting_file.setter
    def setting_file(self,setting):
        self.__setting_file=setting

class Model_setting_parser(Setting_parser):
    def __init__(self):
        super(Model_setting_parser,self).__init__()
    def get_model_settings(self):
        self.config.read(self.setting_file)                
        root="model_setting"
        settings={}
        for k,v in self.config[root].items():
            settings[k]=v
        key_int_value=['total_convolution_layer_size','lstm_layer_number']
        key_float_value=['dropout','learning_rate']
        key_bool_value=['add_terminal_signal','add_batch_normalize']
        key_ints_value=['convolution_layer_sizes','convolution_layer_numbers']
        for key in key_int_value:
            settings[key]=int(settings[key])
        for key in key_float_value:
            settings[key]=float(settings[key])  
        for key in key_bool_value:
            settings[key]=str2bool(settings[key]) 
        for key in key_ints_value:
            settings[key]=[int(i) for i in settings[key].split(",")]
        return settings
class Train_setting_parser(Setting_parser):
    def __init__(self):
        super(Train_setting_parser,self).__init__()
    def get_show_settings(self):
        self.config.read(self.setting_file)                
        show_settings="show"
        settings={}
        key_bool_type=['model','verbose','prompt']
        for key in key_bool_type:
            settings[key]=str2bool(self.config[show_settings][key])
        return settings
    def get_training_settings(self):
        self.config.read(self.setting_file)                           
        training_settings="training_settings"
        settings={}
        key_int_value=['step','progress_target','previous_epoch','batch_size']
        key_array_value=['training_files','validation_files']
        for k,v in self.config[training_settings].items():
            settings[k]=v
        for key in key_int_value:
            settings[key]=int(settings[key])
        for key in key_array_value:
            settings[key]=settings[key].split(",\n")
        return settings
class Batch_ruuning():
    #initialize all the variable needed
    def __init__(self,settings):
        self.__settings=settings
        #initialize all the variable from setting file
        self.__init_parameter()
        #get names and sequences from file
        self.__init_training_files_data()
        self.__init_validation_files_data()
        if self.is_prompt_visible:
            self.print_file_classification()
        #create training and validation set
        self.__init_data()
        if self.is_prompt_visible:
            self.print_selected_information()
        #create model and evaluator
        self.__init_model()
        if self.is_model_visible:
            self.print_model_information()
        self.__init_evaluator()
        self.__restored_previous_result()
        self.__create_stored_folder()
    def __create_stored_folder(self):
        try:
            self.__folder_name=self.__outputfile_root+'/'+str(self.__train_id)+"/"+str(self.__mode_id)
            if not os.path.exists(self.__folder_name):
                print("Create folder:"+self.__folder_name)
                os.makedirs(self.__folder_name)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    def print_model_information(self):
        print("Create model")
        self.model.summary()
    def __restored_previous_result(self):
        if self.__previous_epoch>0:
            if self.is_prompt_visible:
                print("Loading previous status:"+self.__previous_status_root)
            self.__load_previous_result()
        else:
            if self.is_prompt_visible:
                print("Start a new training status")
            return
    def __load_previous_result(self):
        whole_file_path=self.__previous_status_root
        self.evaluator.add_previous_histories(np.load(whole_file_path+'.npy').tolist())
        accuracy_function=tensor_end_with_terminal_binary_accuracy
        binary_crossentropy_function=tensor_end_with_terminal_binary_crossentropy
        self.__model=load_model(whole_file_path+'.h5',custom_objects=
                                     {'tensor_end_with_terminal_binary_accuracy':accuracy_function,
                                      'tensor_end_with_terminal_binary_crossentropy':binary_crossentropy_function})
        self.evaluator.set_model(self.__model)
    def print_file_classification(self):
        print("Status of file:")
        for file in self.__training_files:
            print("\tTraining set:"+file)
        for file in self._Batch_ruuning__validation_files:
            print("\tValidation set:"+file)    
    def __init_training_files_data(self):
        self.__training_seqs=[]
        for file in self.__training_files:
            (name,seq)=fastas2arr(file)
            self.__training_seqs+=seq
    def __init_validation_files_data(self):
        self.__validation_seqs=[]
        for file in self.__validation_files:
            (name,seq)=fastas2arr(file)
            self.__validation_seqs+=seq
    def __init_parameter(self):
        roots=['model','training','show']
        for root in roots:        
            for k,v in self.__settings[root].items():
                variable_tail_name=''
                variable_head_name='_'+self.__class__.__name__+'__'
                if root=='show':
                    variable_tail_name+='_visible'
                    variable_head_name+='is_'
                setattr(self,variable_head_name+k+variable_tail_name, v)
        self.__date=strftime("%Y_%b_%d", gmtime())
        self.__train_id=self.__settings['id']
        self.__setting_record=self.__settings['setting_record']
        self.__image=self.__settings['image']
    @property
    def is_prompt_visible(self):
        return self.__is_prompt_visible
    @property
    def is_model_visible(self):  
        return self.__is_model_visible
    @property
    def is_verbose_visible(self):
        return self.__is_verbose_visible
    def print_training_status(self):
        status={'Progress target':self.__progress_target,'Start progress':self.__previous_epoch,'Step':self.__step,'Batch size':self.__batch_size,'Train id':self.__train_id,'Date':self.__date,'Learning rate':self.__learning_rate}
        print("Training satus:")
        for k,v in status.items():
            print("\t"+k+":"+str(v))
    def __init_model(self):
        convolution_layers_settings=Convolution_layers_settings()
        for i in range(self.__total_convolution_layer_size):
            convolution_layers_settings.add_layer(self.__convolution_layer_numbers[i],self.__convolution_layer_sizes[i])
        model=Exon_intron_finder_factory(convolution_layers_settings.get_settings(),self.__lstm_layer_number,
                                   self.__add_terminal_signal,self.__add_batch_normalize,self.__dropout,self.__learning_rate)    
        self.__model=model
    @property
    def model(self):
        return self.__model
    def __init_data(self):
        (self.__x_train,self.__y_train)=seqs2dnn_data(self.__training_seqs,False)
        (self.__x_validation,self.__y_validation)=seqs2dnn_data(self.__validation_seqs,False)
        self.__training_size=len(self.__training_seqs)
        self.__validation_size=len(self.__validation_seqs)
        self.__valid_training_size=len(self.__y_train)
        self.__valid_validation_size=len(self.__y_validation)
    def get_training_set(self):
        return self.__x_train,self.__y_train
    def get_validation_set(self):
        return self.__x_validation,self.__y_validation
    def __init_evaluator(self):
        self.__evaluator=Model_evaluator()
        self.__evaluator.set_training_data(*self.get_training_set())
        self.__evaluator.set_validation_data(*self.get_validation_set())
        self.__evaluator.set_model(self.model)
    @property
    def evaluator(self):
        return self.__evaluator
    def print_selected_information(self):
        training_size=self.__training_size
        validation_size=self.__validation_size
        valid_training_size=self.__valid_training_size
        valid_validation_size=self.__valid_validation_size
        status={'Selected set number':training_size+validation_size,'Training set number':training_size,'Validation set number':validation_size,'Selected valid set number':valid_training_size+valid_validation_size,'Training valid set number':valid_training_size,'Validation valid set number':valid_validation_size}
        print("Status of data:")
        for k,v in status.items():
            print("\t"+k+":"+str(v))
    def get_whole_path_file(self,progress_number):
        file_name=str(self.__train_id)+'_'+str(self.__mode_id)+'_progress_'+str(progress_number)+'_'
        whole_file_path=self.__folder_name+"/"+file_name+self.__date
        return whole_file_path
    def __prepare_first_train(self):
        saved_new_model=self.get_whole_path_file(0)
        self.model.save(saved_new_model+'.h5')
        key_not_to_store=["model","evaluator","y_validation","training_seqs","x_train","x_validation","settings","y_train","best_model","validation_seqs"]
        if self.is_prompt_visible:
            print('Cretae record file:'+self.__folder_name+"/"+self.__setting_record)
        with open(self.__folder_name+"/"+self.__setting_record,"w") as file:
            writer=csv.writer(file,delimiter=',')
            writer.writerow(["attributes","value"])
            keys=self.__dict__.keys()
            values=self.__dict__.values()
            head="_"+self.__class__.__name__+"__"
            for key in keys:
                body_key=key[len(head):]
                if body_key not in key_not_to_store:
                    writer.writerow([body_key,self.__dict__[key]])
        if self.is_prompt_visible:
            print('Cretae model image:'+self.__folder_name+"/"+self.__image)
        plot_model(self.__model,show_shapes=True, to_file=self.__folder_name+"/"+self.__image)
    def run(self):   
        if self.is_prompt_visible:
            print("Start of running")
        if self.__previous_epoch==0:
            self.__prepare_first_train() 
        for progress in range(self.__previous_epoch,self.__progress_target,self.__step):
            whole_file_path=self.get_whole_path_file(self.__step+progress)
            if self.is_prompt_visible:
                print("Starting training:"+whole_file_path)
            self.evaluator.evaluate(self.__step,self.__batch_size,True,int(self.is_model_visible))
            np.save(whole_file_path+'.npy', self.evaluator.get_histories()) 
            self.model.save(whole_file_path+'.h5')
            if self.is_prompt_visible:
                print("Saved training:"+whole_file_path)
        if self.is_prompt_visible:
            print("End of running")
if __name__=='__main__':
    prompt='batch_running.py --setting=<setting_file>'
    parser = argparse.ArgumentParser(description=prompt)
    parser.add_argument('-t','--train_setting',help='Train setting file name', required=True)
    parser.add_argument('-m','--model_setting',help='Model setting file name', required=True)
    parser.add_argument('-id','--train_id',help='Train id', required=True)
    parser.add_argument('-r','--setting_record',help='File name to save setting', required=True)
    parser.add_argument('-image','--image',help='File name to save image', required=True)
    args = parser.parse_args()
    model_parser=Model_setting_parser()
    model_parser.setting_file=args.model_setting
    train_parser=Train_setting_parser()
    train_parser.setting_file=args.train_setting
    settings={'setting_record':args.setting_record,'image':args.image,'id':args.train_id,'training':train_parser.get_training_settings(),'show':train_parser.get_show_settings(),'model':model_parser.get_model_settings()}
    batch_runner=Batch_ruuning(settings)
    batch_runner.run()
    print("End of program")