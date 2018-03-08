import sys
import os
import argparse
from time import gmtime, strftime, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__+'/..'))))
from sequence_annotation.utils.pipeline import TrainPipeline
from sequence_annotation.model.model_trainer import ModelTrainer
__author__ = 'Ching-Tien Wang'
ANNOTATION_TYPES = ['utr_5', 'utr_3', 'intron', 'cds', 'intergenic_region']
if __name__ == '__main__':
    prompt = 'batch_running.py --setting=<setting_file>'
    parser = argparse.ArgumentParser(description=prompt)
    parser.add_argument('-t','--training_setting_path',help='Train setting file path', required=True)
    parser.add_argument('-m','--model_setting_path',help='Model setting file path', required=True)
    parser.add_argument('-i','--train_id',help='Train id', required=True)
    args = parser.parse_args()
    user_setting = {}
    user_setting['training_setting_path']=args.training_setting_path
    user_setting['model_setting_path']=args.model_setting_path
    user_setting['train_id']=args.train_id
    print('Program start time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    print("User input:"+str(user_setting))
    start_time = time()
    train_pipeline = TrainPipeline(user_setting, ModelTrainer())
    train_pipeline.execute()
    end_time = time()
    time_spend = end_time - start_time
    print('Program end time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    print("Time spent: "+strftime("%H:%M:%S", gmtime(time_spend)))
