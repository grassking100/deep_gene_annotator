import os
import sys
sys.path.append(os.path.abspath(__file__+"/../.."))
import argparse
from sequence_annotation.utils.json_reader import JsonReader
from sequence_annotation.pipeline.pipeline_factory import PipelineFactory
from sequence_annotation.pipeline.batch_pipeline import BatchPipeline
import unittest
import shutil
import pandas as pd
if __name__ == '__main__':
    prompt = 'batch_train.py -t training_setting_path -m model_setting_path -i train_id'
    parser = argparse.ArgumentParser(description=prompt)
    parser.add_argument('-t','--training_setting_path',help='Train setting file path', required=True)
    parser.add_argument('-m','--model_setting_path',help='Model setting file path', required=True)
    parser.add_argument('-i','--train_id',help='Train id', required=True)
    args = parser.parse_args()
    print('Program start time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    start_time = time()  
    user_setting = {}
    work_setting_path=parser.training_setting_path
    model_setting_path=parser.model_setting_path
    train_id=args.train_id
    pipeline = BatchPipeline(data_type='sequence_annotation')
    work_setting = JsonReader().read(work_setting_path)
    model_setting = JsonReader().read(model_setting_path)
    pipeline.execute(train_id,work_setting,model_setting)
    end_time = time()
    time_spend = end_time - start_time
    print('Program end time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    print("Time spent: "+strftime("%H:%M:%S", gmtime(time_spend)))
