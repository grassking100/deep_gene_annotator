import os
import sys
sys.path.append(os.path.abspath(__file__+"/../.."))
import argparse
from time import gmtime, strftime, time
from sequence_annotation.utils.exception import ReturnNoneException
from sequence_annotation.pipeline.train_pipeline import TrainSeqAnnPipeline
from sequence_annotation.worker.model_trainer import ModelTrainer
if __name__ == '__main__':
    prompt = 'batch_running.py --setting=<setting_file>'
    parser = argparse.ArgumentParser(description=prompt)
    parser.add_argument('-t','--training_setting_path',help='Train setting file path', required=True)
    parser.add_argument('-m','--model_setting_path',help='Model setting file path', required=True)
    parser.add_argument('-i','--train_id',help='Train id', required=True)
    args = parser.parse_args()
    print('Program start time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    start_time = time()
    train_pipeline = TrainSeqAnnPipeline(args.train_id,args.training_setting_path,args.model_setting_path)
    train_pipeline.execute()
    end_time = time()
    time_spend = end_time - start_time
    print('Program end time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    print("Time spent: "+strftime("%H:%M:%S", gmtime(time_spend)))
