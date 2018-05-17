import os
import sys
sys.path.append(os.path.abspath(__file__+"/../.."))
import argparse
from time import gmtime, strftime, time
from sequence_annotation.pipeline.pipeline_factory import PipelineFactory
from sequence_annotation.utils.json_reader import JsonReader
if __name__ == '__main__':
    prompt = 'test.py --setting=<setting_file>'
    parser = argparse.ArgumentParser(description=prompt)
    parser.add_argument('-t','--test_setting_path',help='Test setting file path', required=True)
    parser.add_argument('-m','--model_setting_path',help='Model setting file path', required=True)
    parser.add_argument('-i','--test_id',help='Test id', required=True)
    args = parser.parse_args()
    print('Program start time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    start_time = time()
    test_pipeline = PipelineFactory().create('test','sequecne_annotation')
    work_setting = JsonReader().read(args.test_setting_path)
    model_setting = JsonReader().read(args.model_setting_path)
    test_pipeline.execute(args.test_id,work_setting,model_setting)
    end_time = time()
    time_spend = end_time - start_time
    print('Program end time: '+strftime("%Y-%m-%d %H:%M:%S",gmtime()))
    print("Time spent: "+strftime("%H:%M:%S", gmtime(time_spend)))
