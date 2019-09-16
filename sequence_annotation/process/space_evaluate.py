import time
import deepdish as dd
import json
import os
from hyperopt import STATUS_OK
import torch
from torch import nn
from ..genome_handler.load_data import load_data
from ..genome_handler.alt_count import max_alt_count
from ..utils.utils import write_fasta
from .loss import SeqAnnAltLoss,SeqAnnLoss
from .executer import BasicExecutor
from .model import SeqAnnModel,FeatureBlock,RelationBlock,ProjectLayer
from .inference import seq_ann_inference
from .callback import EarlyStop

def _float2int(setting):
    setting_ = {}
    for key,value in setting.items():
        setting_[key] = value
        if isinstance(value,float):
            setting_[key] = int(setting_[key])
    return setting_

def get_executor(setting):
    executor = BasicExecutor()
    if setting['type'] == 'naive':
        pass
    elif setting['type'] == 'hierarchy':
        executor.loss = SeqAnnLoss(**setting['coef'])
        executor.inference = seq_ann_inference
    else:
        raise Exception("Unexcept method: {}".fomrat(setting['type']))
    return executor
        
def get_feature_block(in_channels,setting):
    setting = _float2int(setting)
    setting['cnns_setting'] = _float2int(setting['cnns_setting'])
    block = FeatureBlock(4,**setting)
    return block

def get_relation_block(in_channels,setting):
    setting = _float2int(setting)
    setting['rnns_setting'] = _float2int(setting['rnns_setting'])
    block = RelationBlock(in_channels,**setting)
    return block

def get_projection_layer(in_channels,setting,inference_method):
    inference_type = inference_method['type']
    if inference_type == 'naive':
        out_channels = 3
    elif inference_type == 'hierarchy':
        out_channels = 2
    else:
        raise Exception("Unexcept method: {}".fomrat(inference_type))
    setting = _float2int(setting)
    layer = ProjectLayer(in_channels,out_channels=out_channels,**setting)
    return layer
        
def get_seq_ann_modol(feature_block,relation_block,projection_layer,inference_setting):
    if inference_setting['type'] == 'naive':
        use_sigmoid = False
    elif inference_setting['type'] == 'hierarchy':
        use_sigmoid = True
    else:
        raise Exception("Unexcept method: {}".fomrat(inference_setting['type']))
    model = SeqAnnModel(feature_block,relation_block,projection_layer,
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
        self.batch_size = 32
        self.epoch = 16
        self.augmentation_max = 500
        self.patient= 3
        self.parameter_sharing = False
        
    def objective(self,space):
        self._train_counter +=1
        id_ = 'model_'+str(self._train_counter)
        print("{}: {}".format(id_,space))        
        #Get model and executor        
        executor = get_executor(space['inference_method'])
        feature_block = get_feature_block(space['in_channels'],space['feature_block'])
        relation_block = get_relation_block(feature_block.out_channels,
                                            space['relation_block'])
        projection_layer = get_projection_layer(relation_block.out_channels,
                                                space['projection_layer'],
                                                space['inference_method'])
        model = get_seq_ann_modol(feature_block,relation_block,projection_layer,
                                  space['inference_method'])
        paramter_sharing_setting_path = os.path.join(self._saved_root,'paramter_sharing.json')
        if self.parameter_sharing:
            saved_best = None
            if os.path.exists(paramter_sharing_setting_path):
                with open(paramter_sharing_setting_path, 'r') as fp:
                    parameter_sharing = json.load(fp)
                model_path = parameter_sharing['path']
                saved_best = parameter_sharing[self.eval_target]
                loadable = {}
                saved_state_dict = torch.load(model_path)
                model_state_dict = model.state_dict()
                for key, weights in saved_state_dict.items():
                    if weights.shape == model_state_dict[key].shape:
                        loadable[key] = weights
                model.load_state_dict(loadable, strict=False)
        
        executor.optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
        #Set facade
        path = os.path.join(self._saved_root,id_)
        self.facade.set_root(path,with_test=False)
        self.facade.executor = executor
        ealry_stop = EarlyStop(target=self.eval_target,optimize_min=self.target_min,
                               patient=self.patient,save_best_weights=True,
                               restore_best_weights=True,path=path)
        self.facade.other_callbacks.clean()
        self.facade.other_callbacks.add(ealry_stop)
        #Train
        train_record = self.facade.train(model,batch_size=self.batch_size,
                                         epoch=self.epoch,
                                         augmentation_max=self.augmentation_max)
        #Save record and setting
        target = train_record[self.eval_target]
        best = max(target) if not self.target_min else min(target)
        self.space_result[id_] = {'space':space,self.eval_target:best}
        print("{}: {} --> {}".format(id_,space,best))
        self.records[id_] = train_record
        loss = -best if not self.target_min else best
        if self.parameter_sharing:
            to_save = False
            if saved_best is None:
                to_save = True
            else:
                if self.target_min:
                    if saved_best > best:
                        to_save = True
                else:
                    if saved_best < best:
                        to_save = True
            if to_save:
                model_path = os.path.join(self._saved_root,'paramter_sharing.pth')
                torch.save(model.state_dict(), model_path)
                saved_setting = {'path':model_path,self.eval_target:best,'source':id_}
                with open(paramter_sharing_setting_path,"w") as fp:
                    json.dump(saved_setting,fp)

        return {'loss':loss,'status': STATUS_OK,'eval_time': time.time()}

