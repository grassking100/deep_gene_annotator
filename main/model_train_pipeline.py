import configparser
import argparse
import os, errno
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__+'/..' )) ) )
__author__ = 'Ching-Tien Wang'
def str2bool(value):
    if value=="True":
        return True
    elif value=="False":
        return False
    else:
        assert False, (str)(value)+" is neithor True of False"
class SettingParser(object):
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

class ModelSettingParser(SettingParser):
    def __init__(self):
        super(ModelSettingParser,self).__init__()
    def get_model_settings(self):
        self.config.read(self.setting_file)                
        root="model_setting"
        settings={}
        for k,v in self.config[root].items():
            settings[k]=v
        key_int_value=['total_convolution_layer_size','lstm_layer_number','output_dim']
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
class TrainSettingParser(SettingParser):
    def __init__(self):
        super(TrainSettingParser,self).__init__()
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
        key_array_value=['training_files','validation_files',"training_answers","validation_answers"]
        for k,v in self.config[training_settings].items():
            settings[k]=v
        for key in key_int_value:
            settings[key]=int(settings[key])
        for key in key_array_value:
            settings[key]=settings[key].split(",\n")
        return settings
class ModelTrainPipeline():
    #initialize all the variable needed
    def __init__(self,settings):
        self.__settings=settings
        #initialize all the variable from setting file
        self.__init_parameter()
        #get names and sequences from file
        #create training and validation set
        self.__init_data()
        if self.is_prompt_visible:
            self.print_file_classification()
            self.print_selected_information()
        #create model and trainer
        self.__init_model()
        if self.is_model_visible:
            self.print_model_information()
        self.__init_trainer()
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
        self.trainer.add_previous_histories(np.load(whole_file_path+'.npy').tolist())
        accuracy_function=sequence_annotation.model.model_build_helper.tensor_end_with_terminal_binary_accuracy
        binary_crossentropy_function=sequence_annotation.model.model_build_helper.tensor_end_with_terminal_binary_crossentropy
        self.__model=load_model(whole_file_path+'.h5',custom_objects=
                                     {'tensor_end_with_terminal_binary_accuracy':accuracy_function,
                                      'tensor_end_with_terminal_binary_crossentropy':binary_crossentropy_function})
        self.trainer.set_model(self.__model)
    def print_file_classification(self):
        print("Status of file:")
        for file in self.__training_files:
            print("\tTraining set:"+file)
        for file in self._ModelTrainPipeline__validation_files:
            print("\tValidation set:"+file)    
   
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
        cnn_setting_creator=CnnSettingCreator()
        for i in range(self.__total_convolution_layer_size):
            cnn_setting_creator.add_layer(self.__convolution_layer_numbers[i],self.__convolution_layer_sizes[i])
        model=SeqAnnModelFactory(cnn_setting_creator.get_settings(),self.__lstm_layer_number,self.__add_terminal_signal,self.__add_batch_normalize,self.__dropout,self.__learning_rate,self.__output_dim)    
        self.__model=model
    @property
    def model(self):
        return self.__model
    def __init_data(self):
        self.__x_train=[]
        self.__y_train=[]
        self.__x_validation=[]
        self.__y_validation=[]
        self.__training_size=0
        self.__validation_size=0
        for training_file,training_answer in zip(self.__training_files,self.__training_answers):
            (x,y)=seq_ann_alignment(training_file,training_answer,True)
            self.__x_train+=(x)
            self.__y_train+=(y)
            self.__training_size+=len(x)
        for validation_file,validation_answer in zip(self.__validation_files,self.__validation_answers):
            (x,y)=seq_ann_alignment(validation_file,validation_answer,True)
            self.__x_validation+=(x)
            self.__y_validation+=(y)
            self.__validation_size+=len(x)
    def get_training_set(self):
        return self.__x_train,self.__y_train
    def get_validation_set(self):
        return self.__x_validation,self.__y_validation
    def __init_trainer(self):
        self.__trainer=ModelTrainer()
        (x,y)=self.get_training_set()
        self.__trainer.set_training_data(*self.get_training_set())
        self.__trainer.set_validation_data(*self.get_validation_set())
        self.__trainer.set_model(self.model)
    @property
    def trainer(self):
        return self.__trainer
    def print_selected_information(self):
        training_size=self.__training_size
        validation_size=self.__validation_size
        status={'Selected set number':training_size+validation_size,'Training set number':training_size,'Validation set number':validation_size}
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
        key_not_to_store=["model","trainer","y_validation","training_seqs","x_train","x_validation","settings","y_train","best_model","validation_seqs"]
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
            self.trainer.train(self.__step,self.__batch_size,True,int(self.is_verbose_visible),whole_file_path+'/log/')
            np.save(whole_file_path+'.npy', self.trainer.get_histories()) 
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
    model_parser=ModelSettingParser()
    model_parser.setting_file=args.model_setting
    train_parser=TrainSettingParser()
    train_parser.setting_file=args.train_setting
    settings={'setting_record':args.setting_record,'image':args.image,'id':args.train_id,'training':train_parser.get_training_settings(),'show':train_parser.get_show_settings(),'model':model_parser.get_model_settings()}
    from sequence_annotation.model.model_build_helper import CnnSettingCreator
    from sequence_annotation.model.sequence_annotation_model_factory import SeqAnnModelFactory
    from sequence_annotation.model.model_build_helper import tensor_end_with_terminal_binary_crossentropy
    from sequence_annotation.model.model_build_helper import tensor_end_with_terminal_binary_accuracy
    from sequence_annotation.model.model_trainer import ModelTrainer
    from sequence_annotation.data_handler.DNA_vector import code2vec,codes2vec
    from sequence_annotation.data_handler.training_helper import seqs2dnn_data,seq_ann_alignment,seqs_index_selector
    import random,time,importlib,math,sys,numpy as np
    from time import gmtime,strftime
    from keras.models import load_model
    from keras.utils import plot_model
    import sequence_annotation.model
    import sequence_annotation
    import csv
    batch_runner=ModelTrainPipeline(settings)
    batch_runner.run()
    print("End of program")