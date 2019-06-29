import os
import sys
sys.path.append("./sequence_annotation")
import torch
torch.backends.cudnn.benchmark = True
from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
from sequence_annotation.pytorch.loss import SeqAnnLoss
from sequence_annotation.pytorch.executer import BasicExecutor
from sequence_annotation.pytorch.model import SeqAnnModel,seq_ann_inference,FeatureBlock,RelationBlock
from sequence_annotation.pytorch.customize_layer import Conv1d
from sequence_annotation.data_handler.fasta import write_fasta
from sequence_annotation.pytorch.callback import EarlyStop
from sequence_annotation.utils.load_data import load_data
import json

def train(model,fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,test_chroms,saved_root):
    setting = locals()
    del setting['model']
    with open(os.path.join(saved_root,"main_setting.json"),"w") as outfile:
        json.dump(setting, outfile)
    train_data,val_data,test_data = load_data(fasta_path,ann_seqs_path,max_len,
                                              train_chroms,val_chroms,test_chroms)

    facade = SeqAnnFacade()
    facade.train_seqs,facade.train_ann_seqs = train_data
    facade.val_seqs,facade.val_ann_seqs = val_data[:2]
    facade.test_seqs,facade.test_ann_seqs = test_data
    facade.set_root(saved_root)

    executor = BasicExecutor()
    executor.loss = SeqAnnLoss(intron_coef=1,other_coef=1,nonoverlap_coef=1)
    executor.inference = seq_ann_inference
    facade.executor = executor
    facade.simplify_map = {'gene':['exon','intron'],'other':['other']}
    facade.add_seq_fig(*val_data[2:])
    ealry_stop = EarlyStop(target='val_loss',optimize_min=True,patient=10,
                           save_best_weights=True,restore_best_weights=True,
                           path=saved_root)
    facade.other_callbacks.add(ealry_stop)
                                       
    write_fasta(os.path.join(saved_root,'train.fasta'),facade.train_seqs)
    write_fasta(os.path.join(saved_root,'val.fasta'),facade.val_seqs)
    write_fasta(os.path.join(saved_root,'test.fasta'),facade.test_seqs)
    
    train_record = facade.train(model,batch_size=32,epoch_num=100,augmentation_max=200)
    test_record = facade.test(model,batch_size=32)

if __name__ == '__main__':
    print(sys.argv)
    script_path = sys.argv[0]
    os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]
    fasta_path,ann_seqs_path = sys.argv[2],sys.argv[3]
    max_len = int(sys.argv[4])
    train_chroms = [int(v) for v in sys.argv[5].split(",")]
    val_chroms = [int(v) for v in sys.argv[6].split(",")]
    test_chroms = [int(v) for v in sys.argv[7].split(",")]
    saved_root = sys.argv[8]
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)
    command = 'cp -t {} {}'.format(saved_root,script_path)
    print(command)
    os.system(command)

    feature_block_config={'cnns_setting':{"ln_mode":"after_activation",'num_layers':4,
                                          'out_channels':[16,16,16,16],
                                          'kernel_sizes':[60,60,60,60]}}
    relation_block_config={'rnns_setting':{'num_layers':4,'hidden_size':32,
                                           'batch_first':True,'bidirectional':True},
                           'rnns_type':'GRU'}

    feature_block = FeatureBlock(4,**feature_block_config)
    relation_block = RelationBlock(feature_block.out_channels,**relation_block_config)
    project_layer = Conv1d(relation_block.out_channels,out_channels=2,kernel_size=1)
    model = SeqAnnModel(feature_block,relation_block,project_layer,use_sigmoid=True).cuda()
    with open(os.path.join(saved_root,"model_setting.json"),"w") as outfile:
        json.dump(model.get_config(), outfile)

    train(model,fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,test_chroms,saved_root)