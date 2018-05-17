import os
import sys
sys.path.append(os.path.abspath(__file__+"/../.."))
import argparse
from time import gmtime, strftime, time
from sequence_annotation.pipeline.pipeline_factory import PipelineFactory
from sequence_annotation.utils.json_reader import JsonReader
if __name__ == '__main__':
    prompt = 'train.py --setting=<setting_file>'
    parser = argparse.ArgumentParser(description=prompt)
    parser.add_argument('-t','--training_setting_path',help='Train setting file path', required=True)
    parser.add_argument('-m','--model_setting_path',help='Model setting file path', required=True)
    parser.add_argument('-i','--train_id',help='Train id', required=True)
    args = parser.parse_args()
    print('Program start time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    start_time = time()
    train_pipeline = PipelineFactory().create('train','sequecne_annotation')
    train_pipeline = TrainSeqAnnPipeline()
    work_setting = JsonReader().read(args.training_setting_path)
    model_setting = JsonReader().read(args.model_setting_path)
    train_pipeline.execute(args.train_id,work_setting,model_setting)
    end_time = time()
    time_spend = end_time - start_time
    print('Program end time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    print("Time spent: "+strftime("%H:%M:%S", gmtime(time_spend)))
