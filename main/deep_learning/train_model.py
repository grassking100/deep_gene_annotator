import os
import sys
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
sys.path.append(os.path.dirname(__file__)+"/../..")
from sequence_annotation.utils.utils import create_folder, write_json,copy_path,read_json,read_fasta
from sequence_annotation.utils.utils import BASIC_COLOR_SETTING, BASIC_GENE_ANN_TYPES
from sequence_annotation.genome_handler.ann_seq_processor import class_count
from sequence_annotation.genome_handler.select_data import load_data
from sequence_annotation.process.data_processor import AnnSeqProcessor
from sequence_annotation.process.seq_ann_engine import SeqAnnEngine,check_max_memory_usgae,get_model_executor
from sequence_annotation.process.callback import Callbacks
from sequence_annotation.process.data_generator import SeqDataset,seq_collate_wrapper
from sequence_annotation.process.executor import AdvancedExecutor
from main.utils import backend_deterministic
from main.deep_learning.test_model import test


def _get_max_target_seqs(seqs,ann_seqs,seq_fig_target=None):
    max_count = 0
    selected_id = None
    for ann_seq in ann_seqs:
        count = class_count(ann_seq)[seq_fig_target or 'intron']
        if max_count <= count:
            max_count = count
            selected_id = ann_seq.id
    return seqs[selected_id],ann_seqs[selected_id]

def train(saved_root,epoch,model,executor,train_data,val_data,
          batch_size=None,patient=None,period=None,
          discard_ratio_min=None,discard_ratio_max=None,
          augment_up_max=None,augment_down_max=None,
          deterministic=False,other_callbacks=None,
          concat=False,same_generator=False):
    #Set engine
    engine = SeqAnnEngine(BASIC_GENE_ANN_TYPES)
    engine.batch_size = batch_size
    engine.set_root(saved_root,with_test=False,with_train=True,with_val=True)
    #Add callbacks
    other_callbacks = other_callbacks or Callbacks()
    seq,ann_seq = _get_max_target_seqs(val_data[0],val_data[1])
    seq_fig = engine.get_seq_fig(seq,ann_seq,color_settings=BASIC_COLOR_SETTING)
    other_callbacks.add(seq_fig)
    #Prcoess data
    train_seqs, train_ann_seqs = train_data
    val_seqs, val_ann_seqs = val_data
    raw_data = {
        'training': {'inputs': train_seqs,'answers': train_ann_seqs},
        'validation':{'inputs': val_seqs, 'answers': val_ann_seqs}
    }
    data = engine.process_data(raw_data)
    #Create loader
    collate_fn = seq_collate_wrapper(discard_ratio_min,discard_ratio_max,
                                     augment_up_max,augment_down_max,concat)
    train_gen = engine.create_data_gen(collate_fn,not deterministic)
    if same_generator:
        val_gen = train_gen
    else:
        val_gen = engine.create_basic_data_gen()
    train_loader = train_gen(data['training'])
    val_loader = val_gen(data['validation'])
    #Train
    checkpoint_kwargs={'patient':patient,'period':period}
    worker = engine.train(model,executor,train_loader,val_loader,
                          epoch=epoch,other_callbacks=other_callbacks,
                          checkpoint_kwargs=checkpoint_kwargs)
    return worker

def create_signal_loader(root,ann_seq_processor,batch_size):
    donor_path = os.path.join(root,'donor.fasta')
    acceptor_path = os.path.join(root,'acceptor.fasta')
    fake_donor_path = os.path.join(root,'fake_donor.fasta')
    fake_acceptor_path = os.path.join(root,'fake_acceptor.fasta')
    donor = read_fasta(donor_path)
    acceptor = read_fasta(acceptor_path)
    fake_donor = read_fasta(fake_donor_path)
    fake_acceptor = read_fasta(fake_acceptor_path)
    raw_data = {
        'donor':{'inputs':donor},
        'acceptor':{'inputs':acceptor},
        'fake_donor':{'inputs':fake_donor},
        'fake_acceptor':{'inputs':fake_acceptor}
    }
    data = AnnSeqProcessor(BASIC_GENE_ANN_TYPES).process(raw_data)
    for key,items in data.items():
        data[key]['inputs'] = [torch.FloatTensor(item).transpose(0,1) for item in items['inputs']]

    signal_loader = DataLoader(SeqDataset({'signals':data}),
                               shuffle=True,batch_size=batch_size)
    return signal_loader

def main(saved_root,model_config_path,executor_config_path,
         train_data_path,val_data_path,region_table_path,
         batch_size=None,epoch=None,save_distribution=False,
         model_weights_path=None,executor_weights_path=None,
         deterministic=False,concat=False,splicing_root=None,
         signal_loss_method=None,**kwargs):
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
    
    copied_paths = [model_config_path,executor_config_path,train_data_path,val_data_path]
    executor_config = read_json(executor_config_path)
    #Load, parse and save data
    
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)
    
    resource_path = os.path.join(saved_root,'resource')
    create_folder(resource_path)
    for path in copied_paths:
        copy_path(resource_path,path)
    
    backend_deterministic(deterministic)
    
    #Verify path exist
    if region_table_path is not None and not os.path.exists(region_table_path):
        raise Exception("{} is not exist".format(region_table_path))
    
    temp_model,temp_executor = get_model_executor(model_config_path,executor_config,
                                                  save_distribution=save_distribution)
    
    print("Check memory")
    check_max_memory_usgae(saved_root,temp_model,temp_executor,train_data,
                           val_data,batch_size=batch_size,concat=concat)
    del temp_model
    del temp_executor
    torch.cuda.empty_cache()
    print("Memory is available")
    
   
    if splicing_root is not None:
        train_splicing_root = os.path.join(splicing_root,'train')
        val_splicing_root = os.path.join(splicing_root,'val')
        ann_seq_processor = AnnSeqProcessor(BASIC_GENE_ANN_TYPES)
        train_signal_loader = create_signal_loader(train_splicing_root,ann_seq_processor,batch_size)
        val_signal_loader = create_signal_loader(val_splicing_root,ann_seq_processor,batch_size)
        def wrapper():
            return AdvancedExecutor(train_signal_loader,val_signal_loader,signal_loss_method)
        executor_config['executor_class'] = wrapper

    model,executor = get_model_executor(model_config_path,executor_config,
                                        save_distribution=save_distribution,
                                        model_weights_path=model_weights_path,
                                        executor_weights_path=executor_weights_path)
    
    try:
        train(saved_root,epoch,model,executor,train_data,val_data,
              batch_size=batch_size,deterministic=deterministic,
              concat=concat,**kwargs)     
    except RuntimeError:
        raise Exception("Something wrong ocuurs in {}".format(saved_root))
        
    #Test
    test_paths = []
    data_list = []

    test_paths.append('test_on_train')
    data_list.append(train_data)

    test_paths.append('test_on_val')
    data_list.append(val_data)

    for path,data in zip(test_paths,data_list):
        test_root = os.path.join(saved_root,path)
        create_folder(test_root)
        test(test_root,model,executor,data,batch_size=batch_size,
             region_table_path=region_table_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m","--model_config_path",help="Path of model config "
                        "build by SeqAnnBuilder",required=True)
    parser.add_argument("-e","--executor_config_path",help="Path of Executor config",required=True)
    parser.add_argument("-s","--saved_root",help="Root to save file",required=True)
    parser.add_argument("-t","--train_data_path",help="Path of training data",required=True)
    parser.add_argument("-v","--val_data_path",help="Path of validation data",required=True)
    parser.add_argument("--region_table_path",help="The path of region data table",required=True)
    parser.add_argument("-b","--batch_size",type=int,default=32)
    parser.add_argument("--augment_up_max",type=int,default=0)
    parser.add_argument("--augment_down_max",type=int,default=0)
    parser.add_argument("--discard_ratio_min",type=float,default=0)
    parser.add_argument("--discard_ratio_max",type=float,default=0)
    parser.add_argument("-n","--epoch",type=int,default=100)
    parser.add_argument("-p","--period",default=1,type=int)
    parser.add_argument("--patient",help="The epoch to stop traininig when val_loss "
                        "is not improving. Dafault value is None, the model won't be "
                        "stopped by early stopping",type=int,default=None)
    parser.add_argument("-g","--gpu_id",type=int,default=0,help="GPU to used")
    parser.add_argument("--save_distribution",action='store_true')
    parser.add_argument("--deterministic",action="store_true")
    parser.add_argument("--concat",action="store_true")
    parser.add_argument("--same_generator",action="store_true",
                       help='Use same parameters of training generator to valdation generator')
    parser.add_argument("--model_weights_path")
    parser.add_argument("--executor_weights_path")
    parser.add_argument("--splicing_root")
    parser.add_argument("--signal_loss_method",type=str)
    
    args = parser.parse_args()
    kwargs = dict(vars(args))
    del kwargs['gpu_id']
    
    with torch.cuda.device(args.gpu_id):
        main(**kwargs)
