from Exon_intron_finder.Training_helper import traning_validation_data_index_selector
from Exon_intron_finder.Exon_intron_finder import Convolution_layers_settings,Exon_intron_finder_factory
from Exon_intron_finder.Model_evaluator import Model_evaluator
from DNA_Vector.DNA_Vector import code2vec,codes2vec
from Fasta_handler.Fasta_handler import *
from time import gmtime, strftime
from keras.models import load_model
import Exon_intron_finder
import random
import time
import numpy as np
import importlib
import math


input_files=[]
cross_validation=5
names=[]
seqs=[]
for i in range(cross_validation):
    input_file='Data/seq2017_8_15_include_single_intron_len_1000_5000_inclusive_train_validation_'+str(i+1)+'_of_5.fasta'
    (name,seq)=fastas2arr(input_file)
    names.append(name)
    seqs.append(seq)

train_start_index=[0,2,3,4]
val_index=[1]

tran_seqs=[]
val_seqs=[]
for index in train_start_index:
    tran_seqs+=seqs[index]
for index in val_index:
    val_seqs+=seqs[index]
training_size=len(tran_seqs)
validation_size=len(val_seqs)
print('selected set number:'+(str)(training_size+validation_size))
print('training set number:'+(str)(training_size))
print('validation set number:'+(str)(validation_size))

(x_train,y_train)=seqs2dnn_data(tran_seqs,False)
(x_validation,y_validation)=seqs2dnn_data(val_seqs,False)


valid_training_size=len(y_train)
valid_validation_size=len(y_validation)
print('selected valid set number:'+(str)(valid_training_size+valid_validation_size))
print('training valid set number:'+(str)(valid_training_size))
print('validation valid set number:'+(str)(valid_validation_size))


#create model to be trained
best_convolution=Convolution_layers_settings().add_layer(35,105).add_layer(116,32).add_layer(25,175).get_settings()
best_model=Exon_intron_finder_factory(best_convolution,12,True)
best_model.summary()
evaluator=Model_evaluator()
evaluator.set_training_data(x_train,y_train)
evaluator.set_validation_data(x_validation,y_validation)
evaluator.set_model(best_model)


# In[17]:


progress_target=30
start_progress=0
step=2
batch_size=100
root='Result/train/'
train_id=1
train_file='train_'+str(train_id)+'/'
date='2017_8_16'
mode_id=2


# In[ ]:


for progress in range(start_progress,progress_target,step):
    file_name='train_'+str(train_id)+'_mode_'+str(mode_id)+'_progress_'+str(step+progress)+'_'
    whole_file_path=root+train_file+file_name+date
    print("starting training:"+whole_file_path)
    start=time.time()
    evaluator.evaluate(step,batch_size,True,1)
    end=time.time()
    #print(end-start)
    np.save(whole_file_path+'.npy', evaluator.get_histories()) 
    best_model.save(whole_file_path+'.h5')
    print("saved training:"+whole_file_path)

