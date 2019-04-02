import time
import pandas as pd
from .customize_layer import SeqAnnModel
from hyperopt import STATUS_OK

class SeqAnnParamFinder:
    def __init__(self,model_facade,root):
        self.model_facade = model_facade
        self.affect_length_max = None
        self.eval_target = 'val_macro_F1'
        self.target_max = True
        self._train_counter = 0
        self._root = root
        self.space_result = {}
        self.test_records = {}
        self.records = {}
        
    def objective(self,space):
        self._train_counter +=1
        print(self._train_counter,space)
        for key,val in space.items():
            if key == 'kernel_size':
                space[key]=int(val)
        cnn_num = space['cnns_setting']['layer_num']
        cnn_kernel_size = space['cnns_setting']['kernel_size']
        rnn_num = space['rnns_setting']['layer_num']
        space['rnns_setting']['hidden_sizes']=[space['rnns_setting']['hidden_sizes']]*rnn_num
        space['cnns_setting']['out_channels']=[space['cnns_setting']['out_channels']]*cnn_num
        space['cnns_setting']['kernel_sizes']=[cnn_kernel_size]*cnn_num
        if self.affect_length_max is not None:
            if cnn_num*cnn_kernel_size>=self.affect_length_max:
                raise Exception("CNN with such large number and size will cause sequence not to be complete")
        model = SeqAnnModel(in_channels=4,out_channels=len(self.model_facade.ann_types),
                            init_value=1,**space).cuda()
        id_ = 'model_'+str(self._train_counter)
        train_record = self.model_facade.train(id_,model)
        target = train_record[self.eval_target]
        if self.target_max:
            best = max(target)
        else:
            best = min(target)
        self.space_result[self._train_counter] = {'space':space,self.eval_target:best}
        self.records[self._train_counter] = train_record
        with open(self.root+'/'+id_+"/record.txt","w") as fp:
            fp.write("Space:\n")
            fp.write(str(space))
            fp.write("\nBest "+self.eval_target+":")
            fp.write(str(best))
        if self.model_facade.ratio[2]>0:
            test_records = self.model_facade.test("test_"+str(id_),model)
            with open(self.root+'/test_'+id_+"/record.txt","w") as fp:
                fp.write(str(test_records))
            self.test_records[self._train_counter] = test_records
        if self.target_max:
            return {'loss':-best,'status': STATUS_OK,'eval_time': time.time()}
        else:
            return {'loss':best,'status': STATUS_OK,'eval_time': time.time()}
