import time
from hyperopt import STATUS_OK
import torch
from torch import nn
import deepdish as dd
import json
from ..utils.load_data import load_data
from ..genome_handler.alt_count import max_alt_count
from ..data_handler.fasta import write_fasta
from .customize_layer import SeqAnnModel
from .SA_facade import SeqAnnFacade
from .loss import SeqAnnAltLoss
from .executer import BasicExecutor
from .model import SeqAnnModel,seq_ann_alt_inference,FeatureBlock,RelationBlock,ProjectLayer
from .callback import EarlyStop

def _float2int(setting):
    setting_ = {}
    for key,value in setting_.items():
        setting_[key] = value
        if isinstance(valus,float):
            setting_[key] = int(setting_[key])
    return setting_

def get_executor(self,setting,use_naive=True):
    executor = BasicExecutor()
    if setting['type'] == 'naive':
        pass
    elif setting['type'] == 'hierarchy':
        executor.loss = SeqAnnAltLoss(**setting['coef'])
        executor.inference = seq_ann_alt_inference
    else:
        raise Exception("Unexcept method: {}".fomrat(setting['type']))
        
def get_feature_block(self,in_channels,setting):
    setting = _float2int(setting)
    block = FeatureBlock(4,**setting)
    return block

def get_relation_block(self,in_channels,setting):
    setting = _float2int(setting)
    block = RelationBlock(in_channels,**setting)
    return block

def get_projection_layer(self,in_channels,setting):
    setting = _float2int(setting)
    layer = ProjectLayer(in_channels,**setting)
    return layer
        
def get_seq_ann_modol(feature_block,relation_block,projection_layer,inference_setting):
    if inference_setting['type'] == 'naive':
        use_sigmoid = False
    elif inference_setting['type'] == 'hierarchy':
        use_sigmoid = True
    else:
        raise Exception("Unexcept method: {}".fomrat(inference_setting['type']))
    model = SeqAnnModel(feature_block,relation_block,project_layer,
                        use_sigmoid=use_sigmoid).cuda()
    return model

class SpaceEvaluator:
    def __init__(self,facade,saved_root):
        self.facade = facade
        self._saved_root = saved_root
        self.eval_target = 'val_loss'
        self.target_min = False
        self._train_counter = 0
        self.space_result = {}
        self.records = {}
        self.batch_size=32
        self.epoch_num=16
        self.augmentation_max=500
        self.patient=3

    def objective(self,space):
        self._train_counter +=1
        id_ = 'model_'+str(self._train_counter)
        #Get model and executor
        executor = get_executor(space['inference_method'])
        feature_block = get_feature_block(space['feature_block'])
        relation_block = get_relation_block(feature_block.in_channels,
                                            space['relation_block'])
        projection_layer = get_projection_layer(relation_block.in_channels,
                                                space['projection_layer'])
        seq_ann_modol = get_seq_ann_modol(feature_block,relation_block,projection_
                                          layer,space['inference_method'])
        #Set facade
        path = os.path.join(self._saved_root,id_)
        facade.set_root(path,with_test=False)
        facade.executor = executor
        ealry_stop = EarlyStop(target='val_loss',optimize_min=self.target_min,
                               patient=self.self.patient,save_best_weights=True,
                               restore_best_weights=True,path=path)
        facade.other_callbacks.clean()
        facade.other_callbacks.add(ealry_stop)
        #Train
        train_record = self.facade.train(model,batch_size=self.batch_size,
                                         epoch_num=self.epoch_num,
                                         augmentation_max=self.augmentation_max)
        #Save record and setting
        target = train_record[self.eval_target]
        best = max(target) if self.target_max else min(target)
        self.space_result[self._train_counter] = {'space':space,self.eval_target:best}
        print("{}: {} --> {}".format(id_,space,best))
        self.records[self._train_counter] = train_record
        loss = -best if self.target_max else: best
        return {'loss':loss,'status': STATUS_OK,'eval_time': time.time()}

