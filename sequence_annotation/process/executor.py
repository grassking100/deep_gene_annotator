import warnings
from abc import abstractmethod, ABCMeta, abstractproperty
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ..utils.utils import read_json
from .utils import get_seq_mask, get_name_parameter
from .loss import CCELoss, FocalLoss, LabelLoss, SeqAnnLoss
from .inference import create_basic_inference, seq_ann_inference

OPTIMIZER_CLASS = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'AdamW': optim.AdamW,
    'RMSprop': optim.RMSprop
}


def optimizer_generator(type_,
                        parameters,
                        momentum=None,
                        nesterov=False,
                        amsgrad=False,
                        adam_betas=None,
                        **kwargs):
    momentum = momentum or 0
    adam_betas = adam_betas or [0.9, 0.999]
    if type_ not in OPTIMIZER_CLASS:
        raise Exception("Optimizer should be {}, but got {}".format(
            OPTIMIZER_CLASS, type_))

    optimizer = OPTIMIZER_CLASS[type_]
    if type_ == 'AdamW':
        warnings.warn(
            "\n!!!\n\nAdamW's weight decay is implemented differnetly"
            "in paper and pytorch 1.3.0, please see https://github.com"
            "/pytorch/pytorch/pull/21250#issuecomment-520559993 for more "
            "information!\n\n!!!\n")

    if optimizer in [optim.Adam, optim.AdamW]:
        if momentum > 0 or nesterov:
            warnings.warn(
                "The momentum and nesterov would not be set to Adam and AdamW!!\n"
            )
        return optimizer(parameters,
                         amsgrad=amsgrad,
                         betas=adam_betas,
                         **kwargs)
    elif optimizer in [optim.RMSprop]:
        if nesterov or amsgrad:
            raise Exception("The nesterov and amsgrad are not an option for RMSprop")
        return optimizer(parameters, momentum=momentum, **kwargs)
    else:
        if amsgrad:
            raise Exception("The amsgrad is not an option for this optimizer")
        return optimizer(parameters,
                         momentum=momentum,
                         nesterov=nesterov,
                         **kwargs)


def _evaluate(loss, model,seq_data, inference, **kwargs):
    inputs = seq_data.inputs.cuda()
    labels = seq_data.answers.cuda()
    lengths = seq_data.lengths
    model.train(False)
    with torch.no_grad():
        outputs, lengths = model(inputs, lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths).cuda()
        loss_ = loss(outputs, labels, masks,
                     seq_data=seq_data,**kwargs).item()
        predict_result = inference(outputs, masks)
    return {
        'loss': loss_,
        'predicts': predict_result,
        'lengths': lengths,
        'masks': masks,
        'outputs': outputs.cpu().numpy()
    }


def _predict(model, seq_data, inference):
    inputs = seq_data.inputs.cuda()
    lengths=seq_data.lengths
    model.train(False)
    with torch.no_grad():
        outputs, lengths = model(inputs, lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths).cuda()
        predict_result = inference(outputs, masks)
    return {
        'predicts': predict_result,
        'lengths': lengths,
        'masks': masks,
        'outputs': outputs.cpu().numpy()
    }


class IExecutor(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, model, inputs, labels, lengths, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, model, inputs, labels, lengths, **kwargs):
        pass

    @abstractmethod
    def predict(self, model, inputs, lengths):
        pass

    @abstractmethod
    def get_config(self):
        pass

    @abstractproperty
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dicts):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, metric):
        pass

    @abstractproperty
    def optimizer(self):
        pass

    @optimizer.setter
    @abstractmethod
    def optimizer(self, optimizer):
        pass

    @abstractmethod
    def on_work_begin(self):
        pass


class _Executor(IExecutor):
    def __init__(self):
        self.loss = CCELoss()
        self.inference = create_basic_inference(3)
        self._optimizer = None
        self.clip_grad_value = None
        self.clip_grad_norm = None
        self.grad_norm_type = 2

        self.lr_scheduler = None
        self.lr_scheduler_target = 'val_loss'
        self._has_fit = False
        self._lr_history = {}
       

    def predict(self, model, seq_data):
        model.reset()
        returned = _predict(model, seq_data, inference=self.inference)
        model.reset()
        return returned

    def evaluate(self, model,seq_data, **kwargs):
        model.reset()
        if self.loss is not None:
            returned = _evaluate(self.loss,model,seq_data,
                                 inference=self.inference,**kwargs)
            returned['metric'] = {'loss': returned['loss']}
        else:
            returned = _predict(model,seq_data,
                                inference=self.inference)
            returned['metric'] = {}
        model.reset()
        return returned

    def on_work_begin(self):
        self._has_fit = False
    
    def on_epoch_end(self, epoch, metric):
        self.loss.reset_accumulated_data()
        if self._has_fit:
            for index, group in enumerate(self.optimizer.param_groups):
                if index not in self._lr_history:
                    self._lr_history[index] = []
                self._lr_history[index].append(group['lr'])

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(metric[self.lr_scheduler_target],
                                       epoch=epoch)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
                
    def state_dict(self):
        state_dict = {}
        if self.optimizer is not None:
            state_dict['optimizer'] = self.optimizer.state_dict()
        state_dict['lr_history'] = self._lr_history
        if self.lr_scheduler is not None:
            state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict
      
    def load_state_dict(self, state_dicts):
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state_dicts['optimizer'])
        self._lr_history = state_dicts['lr_history']
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state_dicts['lr_scheduler'])
      
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        if self.loss is not None:
            config['loss_config'] = self.loss.get_config()
        config['inference'] = self.inference.__name__
        config['clip_grad_value'] = self.clip_grad_value
        config['clip_grad_norm'] = self.clip_grad_norm
        config['grad_norm_type'] = self.grad_norm_type
        if self.optimizer is not None:
            config['optimizer_name'] = self.optimizer.__class__.__name__
            param_groups = []
            for group in self.optimizer.state_dict()['param_groups']:
                group = dict(group)
                del group['params']
                param_groups.append(group)
            config['optimizer'] = param_groups
        if self.lr_scheduler is not None:
            config['lr_scheduler_name'] = self.lr_scheduler.__class__.__name__
            config['lr_scheduler'] = self.lr_scheduler.state_dict()
            config['lr_scheduler_target'] = self.lr_scheduler_target
        return config
    
class BasicExecutor(_Executor):
    def fit(self, model, seq_data, **kwargs):
        inputs = seq_data.inputs.cuda()
        labels = seq_data.answers.cuda()
        lengths = seq_data.lengths
        self._has_fit = True
        if self.optimizer is None:
            raise Exception("Exectutor must set optimizer for fitting")
        model.train()
        self.optimizer.zero_grad()
        outputs, lengths = model(inputs, lengths=lengths, answers=labels)
        masks = get_seq_mask(lengths).cuda()
        predict_result = self.inference(outputs, masks)
        loss_ = self.loss(outputs,labels,masks,seq_data=seq_data,**kwargs)
        loss_.backward()
        if self.clip_grad_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), self.clip_grad_value)
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm,
                                     self.grad_norm_type)
        self.optimizer.step()
        returned = {
            'metric': {'loss': loss_.item()},
            'predicts': predict_result,
            'lengths': lengths,
            'masks': masks,
            'outputs': outputs
        }
        return returned

EPSILON=1e-32

def get_signal(model,signals):
    donor = signals['donor']
    acceptor = signals['acceptor']
    #fake_donor = signals['fake_donor']
    #fake_acceptor = signals['fake_acceptor']
    #signals = torch.cat([donor,acceptor,fake_donor,fake_acceptor]).cuda()
    signals = torch.cat([donor,acceptor]).cuda()
    N,C,L = donor.shape
    singal_lengths = [L]*N*2
    if model.norm_input_block is not None:
        signals = model.norm_input_block(signals,singal_lengths)
    signals = model.feature_block(signals,singal_lengths)[0]
    signals = model.relation_block.rnn_1(signals,singal_lengths)
    donor_signals = signals[:N]
    acceptor_signals = signals[N:2*N]
    #fake_donor_signals = signals[2*N:3*N]
    #fake_acceptor_signals = signals[3*N:4*N]
    return donor_signals,acceptor_signals#,fake_donor_signals,fake_acceptor_signals
    
def _calculate_signal_loss(signals,signal_loss_method=None):
    donor_signals,acceptor_signals,fake_donor_signals,fake_acceptor_signals = signals
    donor_loss = -torch.log(donor_signals[:,:,160]+EPSILON).mean() - torch.log(1-fake_donor_signals[:,:,160]+EPSILON).mean()
    acceptor_loss = -torch.log(acceptor_signals[:,:,160]+EPSILON).mean() - torch.log(1-fake_acceptor_signals[:,:,160]+EPSILON).mean()
    signal_loss = donor_loss+acceptor_loss
    return signal_loss

def calculate_signal_loss(signals,signal_loss_method=None):
    #donor_signals,acceptor_signals,fake_donor_signals,fake_acceptor_signals = signals
    donor_signals,acceptor_signals= signals
    N,C,L = donor_signals.shape
    if L%2==0:
        middle = int(L/2)
    else:
        middle = int((L-1)/2)
        
    real_donor_loss = 1-(donor_signals[:,0,middle] - donor_signals[:,0,middle-1]).mean()
    real_acceptor_loss = 1-(acceptor_signals[:,0,middle] - acceptor_signals[:,0,middle+1]).mean()
    #fake_donor_loss = torch.pow(fake_donor_signals[:,0,middle] - fake_donor_signals[:,0,middle-1],2).mean()
    #fake_acceptor_loss = torch.pow(fake_acceptor_signals[:,0,middle] - fake_acceptor_signals[:,0,middle+1],2).mean()
    signal_loss = real_donor_loss+real_acceptor_loss#+fake_donor_loss+fake_acceptor_loss
    return {
        'loss':signal_loss,
        'real_donor_loss':real_donor_loss,
        'real_acceptor_loss':real_acceptor_loss,
       # 'fake_donor_loss':fake_donor_loss,
       # 'fake_acceptor_loss':fake_acceptor_loss
    }
        
class AdvancedExecutor(_Executor):
    def __init__(self,train_signal_loader,val_signal_loader,signal_loss_method):
        super().__init__()
        self.train_signal_loader = train_signal_loader
        self.train_signal_iterator = iter(self.train_signal_loader)
        #self.val_signal_loader = iter(val_signal_loader)
        self.signal_loss_method = signal_loss_method
        
    def fit(self, model, seq_data, **kwargs):
        inputs = seq_data.inputs.cuda()
        labels = seq_data.answers.cuda()
        lengths = seq_data.lengths
        self._has_fit = True
        if self.optimizer is None:
            raise Exception("Exectutor must set optimizer for fitting")
        model.train()
        self.optimizer.zero_grad()
        #Get annotation loss
        outputs, lengths = model(inputs, lengths=lengths, answers=labels)
        masks = get_seq_mask(lengths).cuda()
        predict_result = self.inference(outputs, masks)
        label_loss = self.loss(outputs,labels,masks,seq_data=seq_data,**kwargs)
        #Get similarity loss
        try:
            data = next(self.train_signal_iterator)['signals']
        except StopIteration:
            self.train_signal_iterator = iter(self.train_signal_loader)
            data = next(self.train_signal_iterator)['signals']
        
        signals = get_signal(model,data)
        signal_losses = calculate_signal_loss(signals,self.signal_loss_method)
        signal_loss = signal_losses['loss']
        total_loss = label_loss + signal_loss
        total_loss.backward()
        if self.clip_grad_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), self.clip_grad_value)
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm,
                                     self.grad_norm_type)
        self.optimizer.step()
        returned = {
            'metric': {
                'loss': label_loss.item(),
                'total_loss': total_loss.item(),
                'signal_loss':signal_loss.item(),
                'real_donor_loss':signal_losses['real_donor_loss'].item(),
                'real_acceptor_loss':signal_losses['real_acceptor_loss'].item(),
               # 'fake_donor_loss':signal_losses['fake_donor_loss'].item(),
               # 'fake_acceptor_loss':signal_losses['fake_acceptor_loss'].item(),
            },
            'predicts': predict_result,
            'lengths': lengths,
            'masks': masks,
            'outputs': outputs
        }
        return returned

    
class ExecutorBuilder:
    def __init__(self):
        self.use_native = True
        self._clip_grad_value = None
        self._clip_grad_norm = None
        self._grad_norm_type = 2
        self._gamma = None
        self._intron_coef = None
        self._other_coef = None
        self._answer_label_num = 3
        self._predict_label_num = None
        self._learning_rate = 1e-3
        self._optimizer_kwargs = None
        self._use_lr_scheduler = False
        self._threshold = 0
        self._patience = 10
        self._factor = 0.5
        self._optim_type = 'Adam'
        self._set_optimizer_kwargs = None
        self._set_loss_kwargs = None
        self._set_lr_scheduler_kwargs = None
        self.executor_class = None

    def get_set_kwargs(self):
        config = {}
        config['optimizer_kwargs'] = dict(self._set_optimizer_kwargs)
        config['loss_kwargs'] = dict(self._set_loss_kwargs)
        config['lr_scheduler_kwargs'] = dict(self._set_lr_scheduler_kwargs)
        config['use_native'] = self.use_native
        config['executor_class'] = self.executor_class
        for values in config.values():
            if 'self' in values:
                del values['self']

        return config
        
    def set_optimizer(self,
                      optim_type,
                      learning_rate=None,
                      clip_grad_value=None,
                      clip_grad_norm=None,
                      grad_norm_type=None,
                      **kwargs):
        self._set_optimizer_kwargs = locals()
        self._optimizer_kwargs = kwargs
        self._optim_type = optim_type or self._optim_type
        self._learning_rate = learning_rate or self._learning_rate
        self._clip_grad_value = clip_grad_value or self._clip_grad_value
        self._clip_grad_norm = clip_grad_norm or self._clip_grad_norm
        self._grad_norm_type = grad_norm_type or self._grad_norm_type

    def set_loss(self, gamma=None, intron_coef=None, other_coef=None):
        self._set_loss_kwargs = locals()
        self._gamma = gamma
        self._intron_coef = intron_coef
        self._other_coef = other_coef

    def set_lr_scheduler(self,
                         patience=None,
                         factor=None,
                         threshold=None,
                         use_lr_scheduler=None):
        self._set_lr_scheduler_kwargs = locals()
        if use_lr_scheduler is not None:
            self._use_lr_scheduler = use_lr_scheduler
        self._threshold = threshold or self._threshold
        self._patience = patience or self._patience
        self._factor = factor or self._factor

    def build(self, parameters, executor_weights_path=None):
        optimizer = optimizer_generator(self._optim_type,
                                        parameters,
                                        lr=self._learning_rate,
                                        **self._optimizer_kwargs)
        if self.use_native:
            self._predict_label_num = 3
            self._inference = create_basic_inference(self._predict_label_num)
            loss = FocalLoss(self._gamma)
        else:
            self._predict_label_num = 2
            self._inference = seq_ann_inference
            loss = SeqAnnLoss(intron_coef=self._intron_coef,
                              other_coef=self._other_coef)
            
        if self.executor_class is None:
            exe = BasicExecutor()
        else:
            exe = self.executor_class()
        exe.loss = LabelLoss(loss)
        exe.loss.predict_inference = create_basic_inference(
            self._predict_label_num)
        exe.loss.answer_inference = create_basic_inference(
            self._answer_label_num)
        exe.clip_grad_value = self._clip_grad_value
        exe.clip_grad_norm = self._clip_grad_norm
        exe.grad_norm_type = self._grad_norm_type
        exe.inference = self._inference
        exe.optimizer = optimizer
        if self._use_lr_scheduler:
            exe.lr_scheduler = ReduceLROnPlateau(optimizer,
                                                 verbose=True,
                                                 threshold=self._threshold,
                                                 factor=self._factor,
                                                 patience=self._patience - 1)
        if executor_weights_path is not None:
            weights = torch.load(executor_weights_path)
            exe.load_state_dict(weights)
        return exe


def get_params(model, target_weight_decay=None, weight_decay_name=None):
    target_weight_decay = target_weight_decay or []
    weight_decay_name = weight_decay_name or []
    if len(target_weight_decay) == len(weight_decay_name):
        params = []
        special_names = []
        for name, weight_decay_ in zip(weight_decay_name, target_weight_decay):
            returned_names, parameters = get_name_parameter(model, [name])
            special_names += returned_names
            params += [{'params': parameters, 'weight_decay': weight_decay_}]
        default_parameters = []
        for name_, parameter in model.named_parameters():
            if name_ not in special_names:
                default_parameters.append(parameter)
        params += [{'params': default_parameters}]
        return params
    else:
        raise Exception(
            "Different number between target_weight_decay and weight_decay_name"
        )


def get_executor(model, config_or_path, executor_weights_path=None):
    if isinstance(config_or_path, str):
        config = read_json(config_or_path)
    else:
        config = config_or_path

    builder = ExecutorBuilder()
    builder.use_native = config['use_native']
    if 'executor_class' in config:
        builder.executor_class = config['executor_class']
    builder.set_optimizer(**config['optim_config'])
    builder.set_loss(**config['loss_config'])
    builder.set_lr_scheduler(**config['lr_scheduler_config'])
    if 'weight_decay_config' in config:
        params = get_params(model, **config['weight_decay_config'])
    else:
        params = get_params(model)
    executor = builder.build(params,
                             executor_weights_path=executor_weights_path)
    return executor
