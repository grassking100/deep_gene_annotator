from keras.layers import BatchNormalization, ReLU, Conv1D, Add, Dropout, Concatenate
from keras.models import Sequential
from keras.engine.topology import Layer
class Cnn1dBatchReluBuilder():
    def build(self,conv1d_setting,BN_setting=None,act_name=None,mode=1,name=None):
        """
            Mode 0: CNN --> ReLU --> BN
            Mode 1: CNN --> BN --> ReLU
        """
        if name is not None:
            if "name" not in conv1d_setting.keys():
                conv1d_setting['name'] = name + "_Conv1D"
            if "name" not in BN_setting.keys():
                BN_setting['name'] = name + "_BN"
            if act_name is None:
                act_name = name + "_ReLU"
            if mode in [1]:
                conv1d_setting['use_bias'] = False
            _BN_setting = BN_setting or {}
            if 'kernel_regularizer' in conv1d_setting.keys():
                kernel_regularizer = conv1d_setting['kernel_regularizer']
                exec('from keras.regularizers import {regularizer}'.format(regularizer=kernel_regularizer[:2]))
                exec('conv1d_setting[\'kernel_regularizer\']={regularizer}'.format(regularizer=kernel_regularizer))
        def Cnn1dBatchRelu(inputs):
            conv = Conv1D(**conv1d_setting)
            relu = ReLU(name=act_name)
            BN = BatchNormalization(**_BN_setting)
            if mode == 0:
                return BN(relu(conv(inputs)))
            elif mode == 1:
                return relu(BN(conv(inputs)))
            else:
                raise Exception("Mode "+str(mode)+" has not be supported yet")
        return Cnn1dBatchRelu

class ResidualLayerBuilder():
    def build(self,conv1d_setting,BN_setting=None,act_name=None,add_name=None,
              reshape_name=None,mode=0,dropout_value=0,dropout_name=None,name=None):
        """
            mode 0: BN(ReLU(Add([Conv(inputs),inputs])))
            mode 1: ReLU(BN(Add([Conv(inputs),inputs])))
            mode 2: BN(Add([ReLU(Conv(inputs)),inputs]))
            mode 3: ReLU(Add([BN(Conv(inputs)),inputs]))
            mode 4: Add([BN(ReLU(Conv(inputs))),inputs])
            mode 5: Add([ReLU(BN(Conv(inputs))),inputs])
        """
        if name is not None:
            if "name" not in conv1d_setting.keys():
                conv1d_setting['name'] = name + "_Conv1D"
            if "name" not in BN_setting.keys():
                BN_setting['name'] = name + "_BN"
            if act_name is None:
                act_name = name + "_ReLU"
            if add_name is None:
                add_name = name + "_Add"
            if reshape_name is None:
                reshape_name = name + "_Reshape"
            if dropout_name is None:
                dropout_name = name + "_Dropout"
        if mode in [3,5]:
            conv1d_setting['use_bias'] = False
        else:
            conv1d_setting['use_bias'] = True
        if 'kernel_regularizer' in conv1d_setting.keys():
            kernel_regularizer = conv1d_setting['kernel_regularizer']
            exec('from keras.regularizers import {regularizer}'.format(regularizer=kernel_regularizer[:2]))
            exec('conv1d_setting[\'kernel_regularizer\']={regularizer}'.format(regularizer=kernel_regularizer))
        _BN_setting = BN_setting or {}
        def ResidualLayer(inputs):
            if int(inputs.shape[-1]) != conv1d_setting['filters']:
                reshape = Conv1D(filters=conv1d_setting['filters'],kernel_size=1,padding="same",
                                 name=reshape_name,activation=None)
                same_dim_inputs = reshape(inputs)
            else:
                same_dim_inputs = inputs
            conv = Conv1D(**conv1d_setting)
            relu = ReLU(name=act_name)
            BN = BatchNormalization(**_BN_setting)
            add = Add(name=add_name)
            if mode == 0:
                return BN(relu(add([conv(inputs),same_dim_inputs])))
            elif mode == 1:
                return relu(BN(add([conv(inputs),same_dim_inputs])))
            elif mode == 2:
                return BN(add([relu(conv(inputs)),same_dim_inputs]))
            elif mode == 3:
                return relu(add([BN(conv(inputs)),same_dim_inputs]))
            elif mode == 4:
                after_cnn = BN(relu(conv(inputs)))
                if dropout_value > 0:
                    dropout = Dropout(dropout_value,name=dropout_name)
                    after_cnn = dropout(after_cnn)
                return add([after_cnn,same_dim_inputs])
            elif mode == 5:
                after_cnn = relu(BN(conv(inputs)))
                if dropout_value > 0:
                    dropout = Dropout(dropout_value,name=dropout_name)
                    after_cnn = dropout(after_cnn)
                return add([after_cnn,same_dim_inputs])
            else:
                raise Exception("Mode "+str(mode)+" has not be supported yet")
        return ResidualLayer
class DeepResidualLayerBuilder():
    def build(self,kernel_size_filters_tuples,conv1d_setting,BN_setting=None,
              act_name=None,add_name=None,reshape_name=None,mode=0,
              dropout_value=0,dropout_name=None,name=None):
        """
            mode 0: BN(ReLU(Add([Conv(inputs),inputs])))
            mode 1: ReLU(BN(Add([Conv(inputs),inputs])))
            mode 2: BN(Add([ReLU(Conv(inputs)),inputs]))
            mode 3: ReLU(Add([BN(Conv(inputs)),inputs]))
            mode 4: Add([BN(ReLU(Conv(inputs))),inputs])
            mode 5: Add([ReLU(BN(Conv(inputs))),inputs])
        """
        def DeepResidualLayer(inputs):
            previous_layer = inputs
            index = 1
            for kernel_size, filters in kernel_size_filters_tuples:
                conv1d_setting['filters'] = filters
                conv1d_setting['kernel_size'] = kernel_size
                builder = ResidualLayerBuilder()
                if name is not None:
                    sublayer_name = name+"_"+str(index)
                else:
                    sublayer_name = None
                residual = builder.build(dict(conv1d_setting),dict(BN_setting),
                                         act_name,add_name,reshape_name,
                                         mode,dropout_value,dropout_name,
                                         sublayer_name)
                index += 1
                previous_layer = residual(previous_layer)
            return previous_layer
        return DeepResidualLayer    
    
class DenseLayerBuilder():
    def build(self,conv1d_setting,BN_setting=None,act_name=None,concat_name=None,
              mode=0,dropout_value=0,dropout_name=None,name=None):
        """
            mode 0: BN(ReLU(concat([Conv(inputs),inputs])))
            mode 1: ReLU(BN(concat([Conv(inputs),inputs])))
            mode 2: BN(concat([ReLU(Conv(inputs)),inputs]))
            mode 3: ReLU(concat([BN(Conv(inputs)),inputs]))
            mode 4: concat([BN(ReLU(Conv(inputs))),inputs])
            mode 5: concat([ReLU(BN(Conv(inputs))),inputs])
        """
        if name is not None:
            if "name" not in conv1d_setting.keys():
                conv1d_setting['name'] = name + "_Conv1D"
            if "name" not in BN_setting.keys():
                BN_setting['name'] = name + "_BN"
            if act_name is None:
                act_name = name + "_ReLU"
            if concat_name is None:
                concat_name = name + "_Concat"
            if dropout_name is None:
                dropout_name = name + "_Dropout"
        if mode in [3,5]:
            conv1d_setting['use_bias'] = False
        else:
            conv1d_setting['use_bias'] = True
        if 'kernel_regularizer' in conv1d_setting.keys():
            kernel_regularizer = conv1d_setting['kernel_regularizer']
            exec('from keras.regularizers import {regularizer}'.format(regularizer=kernel_regularizer[:2]))
            exec('conv1d_setting[\'kernel_regularizer\']={regularizer}'.format(regularizer=kernel_regularizer))
        _BN_setting = BN_setting or {}
        def DenseLayer(inputs):
            conv = Conv1D(**conv1d_setting)
            relu = ReLU(name=act_name)
            BN = BatchNormalization(**_BN_setting)
            concat = Concatenate(name=concat_name)
            if mode == 0:
                return BN(relu(concat([conv(inputs),inputs])))
            elif mode == 1:
                return relu(BN(concat([conv(inputs),inputs])))
            elif mode == 2:
                return BN(concat([relu(conv(inputs)),inputs]))
            elif mode == 3:
                return relu(concat([BN(conv(inputs)),inputs]))
            elif mode == 4:
                after_cnn = BN(relu(conv(inputs)))
                if dropout_value > 0:
                    dropout = Dropout(dropout_value,name=dropout_name)
                    after_cnn = dropout(after_cnn)
                return concat([after_cnn,inputs])
            elif mode == 5:
                after_cnn = relu(BN(conv(inputs)))
                if dropout_value > 0:
                    dropout = Dropout(dropout_value,name=dropout_name)
                    after_cnn = dropout(after_cnn)
                return concat([after_cnn,inputs])
            else:
                raise Exception("Mode "+str(mode)+" has not be supported yet")
        return DenseLayer    
    
class DeepDenseLayerBuilder():
    def build(self,kernel_size_filters_tuples,conv1d_setting,BN_setting=None,
              act_name=None,concat_name=None,mode=0,dropout_value=0,
              dropout_name=None,name=None):
        """
            mode 0: BN(ReLU(concat([Conv(inputs),inputs])))
            mode 1: ReLU(BN(concat([Conv(inputs),inputs])))
            mode 2: BN(concat([ReLU(Conv(inputs)),inputs]))
            mode 3: ReLU(concat([BN(Conv(inputs)),inputs]))
            mode 4: concat([BN(ReLU(Conv(inputs))),inputs])
            mode 5: concat([ReLU(BN(Conv(inputs))),inputs])
        """
        def DeepDenseLayer(inputs):
            previous_layer = inputs
            index = 1
            for kernel_size, filters in kernel_size_filters_tuples:
                conv1d_setting['filters'] = filters
                conv1d_setting['kernel_size'] = kernel_size
                builder = DenseLayerBuilder()
                if name is not None:
                    sublayer_name = name+"_"+str(index)
                else:
                    sublayer_name = None
                residual = builder.build(dict(conv1d_setting),dict(BN_setting),
                                         act_name,concat_name,mode,
                                         dropout_value,dropout_name,
                                         sublayer_name)
                index += 1
                previous_layer = residual(previous_layer)
            return previous_layer
        return DeepDenseLayer