import os
import sys
sys.path.append("./sequence_annotation")
import torch
torch.backends.cudnn.benchmark = True
from sequence_annotation.pytorch.SA_facade import SeqAnnFacade
from sequence_annotation.pytorch.loss import SeqAnnAltLoss
from sequence_annotation.pytorch.executer import BasicExecutor
from sequence_annotation.pytorch.model import SeqAnnModel,seq_ann_alt_inference,FeatureBlock,RelationBlock,ProjectLayer,HierachyRNN
from sequence_annotation.data_handler.fasta import write_fasta
from sequence_annotation.pytorch.callback import EarlyStop
from sequence_annotation.utils.load_data import load_data
from sequence_annotation.genome_handler.alt_count import max_alt_count
from torch import nn
import deepdish as dd
import json

def train(model,fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,
          test_chroms,saved_root,use_naive):
    setting = locals()
    del setting['model']
    setting_path = os.path.join(saved_root,"main_setting.json")
    with open(setting_path,"w") as outfile:
        json.dump(setting, outfile)
    use_alt=True
    before_mix_simplify_map={'exon':['utr_5','utr_3','cds'],
                             'intron':['intron'],'other':['other']}

    simplify_map={'exon':['exon'],'intron':['intron'],
                  'other':['other'],'alternative':['exon_intron']}
    gene_map = {'gene':['exon','intron','alternative'],'other':['other']}
    print("Load and parse data")
    data_path = os.path.join(saved_root,"data.h5")
    if os.path.exists(data_path):    
        data = dd.io.load(data_path)
    else:    
        data = load_data(fasta_path,ann_seqs_path,max_len,
                         train_chroms,val_chroms,test_chroms,simplify_map,
                         before_mix_simplify_map=before_mix_simplify_map)
        dd.io.save(data_path,data)

    train_data,val_data,test_data = data
    if use_alt:
        alt_num = max_alt_count(train_data[1],gene_map)
        val_alt_num = max_alt_count(val_data[1],gene_map)
        print("Training data's maximmum alternative regions number is {}".format(alt_num))
        print("Validation data's maximmum alternative regions number is {}".format(val_alt_num))
        if len(test_data[0])>0:
            test_alt_num = max_alt_count(test_data[1],gene_map)
            print("Testing data's maximmum alternative regions number is {}".format(test_alt_num))
    
    facade = SeqAnnFacade()
    
    facade.use_gffcompare = False
    facade.alt = use_alt
    if use_alt:
        print("Facade will use {} as its maximmum alternative regions".format(alt_num))
        facade.alt_num=alt_num
    facade.train_seqs,facade.train_ann_seqs = train_data
    print(facade.train_ann_seqs.ANN_TYPES)
    facade.val_seqs,facade.val_ann_seqs = val_data[:2]
    if len(test_data[0])>0:
        facade.test_seqs,facade.test_ann_seqs = test_data
    facade.set_root(saved_root)

    executor = BasicExecutor()
    if not use_naive:
        executor.loss = SeqAnnAltLoss(intron_coef=1,other_coef=1,alt_coef=1)
        executor.inference = seq_ann_alt_inference
    facade.executor = executor
    facade.simplify_map = gene_map
    color_settings={'other':'blue','exon':'red','intron':'yellow',"alternative":"green"}
    facade.add_seq_fig(*val_data[2:],color_settings=color_settings)
    ealry_stop = EarlyStop(target='val_loss',optimize_min=True,patient=3,
                           save_best_weights=True,restore_best_weights=True,
                           path=saved_root)
    facade.other_callbacks.add(ealry_stop)
                                       
    write_fasta(os.path.join(saved_root,'train.fasta'),facade.train_seqs)
    write_fasta(os.path.join(saved_root,'val.fasta'),facade.val_seqs)
    if len(test_data[0])>0:
        write_fasta(os.path.join(saved_root,'test.fasta'),facade.test_seqs)
    
    train_record = facade.train(model,batch_size=32,epoch_num=16,augmentation_max=200)
    if len(test_data[0])>0:
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
    use_naive = sys.argv[9]=='T'
    use_hierarchy = sys.argv[10]=='T'
    if not os.path.exists(saved_root):
        os.mkdir(saved_root)
    command = 'cp -t {} {}'.format(saved_root,script_path)
    print(command)
    os.system(command)    
    feature_block_config={'cnns_setting':{"norm_mode":"after_activation",
                                          'out_channels':16,'kernel_size':60},'num_layers':4}
    feature_block = FeatureBlock(4,**feature_block_config)
    
    if use_naive:
        out_channels = 4
        use_sigmoid=False
        relation_block_config={'rnns_setting':{'num_layers':8,'hidden_size':32,
                                               'batch_first':True,'bidirectional':True},
                               'rnns_type':nn.GRU}
        relation_block = RelationBlock(feature_block.out_channels,**relation_block_config)
    else:    
        out_channels = 3
        use_sigmoid=True
        if use_hierarchy:
            relation_block_config={'rnn_setting':{'num_layers':1,'hidden_size':32,
                                                  'batch_first':True,'bidirectional':True},
                                   'rnns_type':nn.GRU,'num_layers':8}
            relation_block = HierachyRNN(feature_block.out_channels,**relation_block_config)
        else:    
            relation_block_config={'rnns_setting':{'num_layers':8,'hidden_size':32,
                                                   'batch_first':True,'bidirectional':True},
                                   'rnns_type':nn.GRU}
            relation_block = RelationBlock(feature_block.out_channels,**relation_block_config)
    project_layer = ProjectLayer(relation_block.out_channels,out_channels)
    model = SeqAnnModel(feature_block,relation_block,project_layer,use_sigmoid=use_sigmoid).cuda()
    with open(os.path.join(saved_root,"model_setting.json"),"w") as outfile:
        json.dump(model.get_config(), outfile)

    train(model,fasta_path,ann_seqs_path,max_len,train_chroms,val_chroms,test_chroms,saved_root,use_naive)