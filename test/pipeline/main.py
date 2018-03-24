import os
import sys
sys.path.append('/home/grassking100/deep_learning/keras')
sys.path.append(os.path.abspath("~/../../.."))
import argparse
from time import gmtime, strftime, time
from sequence_annotation.pipeline.train_pipeline import TrainPipeline
from sequence_annotation.worker.model_trainer import ModelTrainer
user_setting = {}
user_setting['work_setting_path']='setting/training_setting.json'
user_setting['model_setting_path']='setting/model_setting.json'
user_setting['train_id']='test_03'
print('Program start time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
print("User input:"+str(user_setting))
start_time = time()
train_pipeline = TrainPipeline("pipeline_test",user_setting['work_setting_path'],user_setting['model_setting_path'])
train_pipeline.execute()
end_time = time()
time_spend = end_time - start_time
print('Program end time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
print("Time spent: "+strftime("%H:%M:%S", gmtime(time_spend)))