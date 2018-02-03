import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__+'/..'))))
from sequence_annotation.utils.pipeline import TrainPipeline
from sequence_annotation.model.model_trainer import ModelTrainer

__author__ = 'Ching-Tien Wang'
ANNOTATION_TYPES = ['utr_5', 'utr_3', 'intron', 'cds', 'intergenic_region']
if __name__ == '__main__':
    prompt = 'batch_running.py --setting=<setting_file>'
    #parser = argparse.ArgumentParser(description=prompt)
    #parser.add_argument('-t','--train_setting',help='Train setting file name', required=True)
    #parser.add_argument('-m','--model_setting',help='Model setting file name', required=True)
    #parser.add_argument('-id','--train_id',help='Train id', required=True)
    #parser.add_argument('-r','--setting_record',help='File name to save setting', required=True)
    #parser.add_argument('-image','--image',help='File name to save image', required=True)
    #args = parser.parse_args()
    user_setting = {'setting_record_path':'save.csv',
                    'image_path':'test.png',
                    'train_id':'my_test_2018_1_24',
                    'training':'training_setting_mode_1.ini',
                    'show':'image',
                    'model':'best_model_setting_expanding.ini'}
    train_pipeline = TrainPipeline(user_setting, ModelTrainer())
    train_pipeline.execute()
    print("End of program")
