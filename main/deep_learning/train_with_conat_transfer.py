import os
import sys
import torch
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, read_json, write_json
from main.deep_learning.train_model import main as train_main

def main(saved_root,warmup_epoch=None,
         discard_ratio_min=None,discard_ratio_max=None,
         augment_up_max=None,augment_down_max=None,
         source_status=None,**kwargs):
    source_status = source_status or 'native'
    setting = locals()
    kwargs = setting['kwargs']
    del setting['kwargs']
    setting.update(kwargs)
    #Create folder
    create_folder(saved_root)
    #Save setting
    setting_path = os.path.join(saved_root,"main_setting.json")
    if os.path.exists(setting_path):
        existed = read_json(setting_path)
        if setting != existed:
            raise Exception("The {} is not same as previous one".format(setting_path))
    else:
        write_json(setting,setting_path)
    
    without_concat_root = os.path.join(saved_root,'without_concat')
    #Train without concat
    if source_status == 'native':
        train_main(without_concat_root,concat=False,**kwargs)
    elif source_status == 'aug':
        train_main(without_concat_root,concat=False,
                   discard_ratio_min=discard_ratio_min,
                   discard_ratio_max=discard_ratio_max,
                   augment_up_max=augment_up_max,
                   augment_down_max=augment_down_max,
                   **kwargs)
    else:
        raise

    #Train with concat by transfer
    transfer_root = os.path.join(saved_root,'transfer')
    best_native_weights = os.path.join(without_concat_root,'checkpoint','best_model.pth')
    train_main(transfer_root,concat=True,model_weights_path=best_native_weights,
               discard_ratio_min=discard_ratio_min,discard_ratio_max=discard_ratio_max,
               augment_up_max=augment_up_max,augment_down_max=augment_down_max,
               warmup_epoch=warmup_epoch,**kwargs)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config "
                        "build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("--val_data_for_test_path",help="Path of validation data for testing")
    parser.add_argument("--region_table_path",help="The path of region data table",required=True)
    parser.add_argument("-b","--batch_size",type=int,default=32)
    parser.add_argument("--augment_up_max",type=int,default=0)
    parser.add_argument("--augment_down_max",type=int,default=0)
    parser.add_argument("--discard_ratio_min",type=float,default=0)
    parser.add_argument("--discard_ratio_max",type=float,default=0)
    parser.add_argument("-n","--epoch",type=int,default=100)
    parser.add_argument("--period",default=1,type=int)
    parser.add_argument("--warmup_epoch",default=0,type=int)
    parser.add_argument("--patience",help="The epoch to stop traininig when val_loss "
                        "is not improving. Dafault value is None, the model won't be "
                        "stopped by early stopping",type=int,default=None)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--deterministic",action="store_true")
    parser.add_argument("--same_generator",action="store_true",
                       help='Use same parameters of training generator to valdation generator')
    #parser.add_argument("--model_weights_path")
    #parser.add_argument("--executor_weights_path")
    parser.add_argument("--splicing_root")
    parser.add_argument("--drop_last",action="store_true")
    parser.add_argument("--signal_loss_method",type=str)
    parser.add_argument("--source_status",type=str,default='native')
    
    args = parser.parse_args()
    kwargs = dict(vars(args))
    del kwargs['gpu_id']
    
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
