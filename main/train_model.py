import os
import sys
import json
import torch
import deepdish as dd
from argparse import ArgumentParser
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
from sequence_annotation.utils.utils import create_folder, write_fasta, write_json
from sequence_annotation.utils.utils import BASIC_GENE_MAP,BASIC_GENE_ANN_TYPES
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine
from sequence_annotation.process.convert_signal_to_gff import build_ann_vec_gff_converter
from sequence_annotation.process.utils import param_num
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from main.utils import get_model, get_executor,BASIC_COLOR_SETTING, copy_path,load_data
from main.test_model import test

def _get_max_target_seqs(seqs,ann_seqs,seq_fig_target=None):
    max_count = 0
    selected_id = None
    for ann_seq in ann_seqs:
        count = class_count(ann_seq)[seq_fig_target or 'intron']
        if max_count <= count:
            max_count = count
            selected_id = ann_seq.id
    return seqs[selected_id],ann_seqs[selected_id]

def train(saved_root,epoch,model,executor,train_data,val_data=None,
          batch_size=None,augmentation_max=None,patient=None,
          monitor_target=None,period=None):
    with open(os.path.join(saved_root,'param_num.txt'),"w") as fp:
        fp.write("Required-gradient parameters number:{}".format(param_num(model)))
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.set_root(saved_root,with_test=False,with_train=True,
                    with_val=val_data is not None)
    other_callbacks = Callbacks()
    if val_data is not None:
        seq,ann_seq = _get_max_target_seqs(val_data[0],val_data[1])
        seq_fig = engine.get_seq_fig(seq,ann_seq,color_settings=BASIC_COLOR_SETTING)
        other_callbacks.add(seq_fig)

    worker = engine.train(model,executor,train_data,val_data=val_data,
                          batch_size=batch_size,epoch=epoch,
                          augmentation_max=augmentation_max,
                          other_callbacks=other_callbacks,
                          checkpoint_kwargs={'monitor_target':monitor_target,
                                             'patient':patient,'period':period})
    return worker

def main(saved_root,model_config_path,executor_config_path,
         train_data_path,val_data_path=None,test_data_path=None,
         batch_size=None,epoch=None,
         model_weights_path=None,executor_weights_path=None,
         frozen_names=None,save_distribution=False,
         only_train=False,
         train_answer_path=None,#train_region_path=None,
         val_answer_path=None,#val_region_path=None,
         test_answer_path=None,#test_region_path=None,
         region_table_path=None,**kwargs):
    
    #Load, parse and save data
    train_data = load_data(train_data_path)
    write_fasta(os.path.join(saved_root,'train.fasta'),train_data[0])
    val_data = test_data = None
    if val_data_path is not None:
        val_data = load_data(val_data_path)
        write_fasta(os.path.join(saved_root,'val.fasta'),val_data[0])
        
    if test_data_path is not None:
        test_data = load_data(test_data_path)
        write_fasta(os.path.join(saved_root,'test.fasta'),test_data[0])
    data = train_data, val_data, test_data
    data_path = os.path.join(saved_root,"data.h5")
    dd.io.save(data_path,data)
    
    #Verify path exist
    paths = [train_answer_path,val_answer_path,
             test_answer_path,region_table_path,
            # train_region_path,val_region_path,
#             test_region_path
            ]

    for path in paths:
        if path is not None:
            if not os.path.exists(path):
                raise Exception("{} is not exist".format(path))
    
    #Create model
    
    model = get_model(model_config_path,model_weights_path,frozen_names)
    model.save_distribution = save_distribution

    #Create executor
    with open(executor_config_path,"r") as fp:
        executor_config = json.load(fp)
    
    executor = get_executor(model,executor_weights_path=executor_weights_path,**executor_config)

    try:
        train(saved_root,epoch,model,executor,train_data,val_data,
              batch_size=batch_size,**kwargs)     
    except RuntimeError:
        raise Exception("Something wrong ocuurs in {}".format(saved_root))
        
    #Test
    if not only_train:
        ann_vec_gff_converter = build_ann_vec_gff_converter(BASIC_GENE_ANN_TYPES,BASIC_GENE_MAP)
        executor = get_executor(model,set_loss=False,set_optimizer=False,**executor_config)
        test_paths = ['test_on_train']
        data_list = [train_data]
        answer_paths = [train_answer_path]
        #region_paths = [train_region_path]
        
        if val_data_path is not None:
            test_paths.append('test_on_val')
            data_list.append(val_data)
            answer_paths.append(val_answer_path)
            #region_paths.append(val_region_path)

        if test_data_path is not None:
            test_paths.append('test_on_test')
            data_list.append(test_data)
            answer_paths.append(test_answer_path)
            #region_paths.append(test_region_path)
            
        for path,data,answer_path in zip(test_paths,data_list,answer_paths):
            path = os.path.join(saved_root,path)
            create_folder(path)
            test(path,model,executor,data,
                 batch_size=batch_size,use_gffcompare=True,
                 ann_vec_gff_converter=ann_vec_gff_converter,
                 region_table_path=region_table_path,
                 answer_gff_path=answer_path,
                # answer_region_path=region_path
                )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config "
                        "build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data")
    parser.add_argument("-x","--test_data_path",help="Path of testing data")
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--augmentation_max",type=int,default=0)
    parser.add_argument("--epoch",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--period",default=5,type=int)
    parser.add_argument("--model_weights_path")
    parser.add_argument("--executor_weights_path")
    parser.add_argument("--only_train",action='store_true')
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--patient",help="Dafault value is 5. If lower(patient) "
                        "is 'none', then model won't be stopped",
                        type=lambda x: int(x) if x.lower() != 'none' else None,default=5)
    parser.add_argument("--frozen_names",type=lambda x:x.split(','),default=None)
    parser.add_argument("--monitor_target")
    parser.add_argument("--region_table_path")
    parser.add_argument("--train_answer_path",help='The training answer in gff fomrat')
    parser.add_argument("--val_answer_path",help='The validate answer in gff fomrat')
    parser.add_argument("--test_answer_path",help='The testing answer in gff fomrat')
    #parser.add_argument("--train_region_path",help='The training region ids')
    #parser.add_argument("--val_region_path",help='The validate answer region ids')
    #parser.add_argument("--test_region_path",help='The testing answer region ids')

    args = parser.parse_args()
                
    #Create folder
    create_folder(args.saved_root)
    #Save setting
    setting_path = os.path.join(args.saved_root,"main_setting.json")
    setting = vars(args)
    write_json(setting,setting_path)

    copy_path(args.saved_root,sys.argv[0])
    copy_path(args.saved_root,args.model_config_path)
    copy_path(args.saved_root,args.executor_config_path)

    kwargs = dict(setting)
    del kwargs['saved_root']
    del kwargs['model_config_path']
    del kwargs['executor_config_path']
    del kwargs['gpu_id']
    
    with torch.cuda.device(args.gpu_id):
        main(args.saved_root,args.model_config_path,
             args.executor_config_path,**kwargs)
