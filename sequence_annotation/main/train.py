import sys
import os
import argparse
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
    print(user_setting)
    """user_setting = {'setting_record_path':'save.csv',
                    'image_path':'test.png',
                    'train_id':'my_test_2018_1_24',
                    'training_setting_file':'deep_learning/training_setting_mode_1.ini',
                    'show':'image',
                    'model_setting_file':'deep_learning/best_model_setting_expanding.ini'}"""
    train_pipeline = TrainPipeline(user_setting, ModelTrainer())
    train_pipeline.execute()
    print("End of program")
