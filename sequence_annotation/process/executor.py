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


def _evaluate(loss, model, inputs, labels, lengths, inference, **kwargs):
    model.train(False)
    with torch.no_grad():
        outputs, lengths = model(inputs, lengths=lengths)
        outputs = outputs.float()
        masks = get_seq_mask(lengths).cuda()
        loss_ = loss(outputs, labels, masks, **kwargs).item()
        predict_result = inference(outputs, masks)
    return {
        'loss': loss_,
        'predicts': predict_result,
        'lengths': lengths,
        'masks': masks,
        'outputs': outputs.cpu().numpy()
    }


def _predict(model, inputs, lengths, inference):
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

    def get_config(self):
        config = {}
        if self.loss is not None:
            config['loss_config'] = self.loss.get_config()
        config['inference'] = self.inference.__name__
        return config

    def predict(self, model, inputs, lengths):
        inputs = inputs.cuda()
        returned = _predict(model, inputs, lengths, inference=self.inference)
        return returned

    def evaluate(self, model, inputs, labels, lengths, **kwargs):
        inputs = inputs.cuda()
        labels = labels.cuda()
        if self.loss is not None:
            returned = _evaluate(self.loss,
                                 model,
                                 inputs,
                                 labels,
                                 lengths,
                                 inference=self.inference,
                                 **kwargs)
            returned['metric'] = {'loss': returned['loss']}
        else:
            returned = _predict(model,
                                inputs,
                                lengths,
                                inference=self.inference)
            returned['metric'] = {}
        return returned


class BasicExecutor(_Executor):
    def __init__(self):
        super().__init__()
        self.clip_grad_value = None
        self.clip_grad_norm = None
        self.grad_norm_type = 2
        self._optimizer = None
        self.lr_scheduler = None
        self.lr_scheduler_target = 'val_loss'
        self._has_fit = False
        self._lr_history = {}

    def on_work_begin(self):
        self._has_fit = False

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
        config = super().get_config()
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

    def fit(self, model, inputs, labels, lengths, **kwargs):
        self._has_fit = True
        if self.optimizer is None:
            raise Exception("Exectutor must set optimizer for fitting")
        inputs = inputs.cuda()
        labels = labels.cuda()
        model.train()
        self.optimizer.zero_grad()
        outputs, lengths = model(inputs, lengths=lengths, answers=labels)
        masks = get_seq_mask(lengths).cuda()
        predict_result = self.inference(outputs, masks)
        loss_ = self.loss(outputs,
                          labels,
                          masks,
                          predict_result=predict_result,
                          **kwargs)
        loss_.backward()
        if self.clip_grad_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), self.clip_grad_value)
        if self.clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm,
                                     self.grad_norm_type)
        self.optimizer.step()
        returned = {
            'metric': {
                'loss': loss_.item()
            },
            'predicts': predict_result,
            'lengths': lengths,
            'masks': masks,
            'outputs': outputs
        }
        return returned

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


class ExecutorBuilder:
    def __init__(self, use_native=True):
        self.use_native = use_native
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

    def set_optimizer(self,
                      optim_type,
                      learning_rate=None,
                      clip_grad_value=None,
                      clip_grad_norm=None,
                      grad_norm_type=None,
                      **kwargs):
        self._optimizer_kwargs = kwargs
        self._optim_type = optim_type or self._optim_type
        self._learning_rate = learning_rate or self._learning_rate
        self._clip_grad_value = clip_grad_value or self._clip_grad_value
        self._clip_grad_norm = clip_grad_norm or self._clip_grad_norm
        self._grad_norm_type = grad_norm_type or self._grad_norm_type

    def set_loss(self, gamma=None, intron_coef=None, other_coef=None):
        self._gamma = gamma
        self._intron_coef = intron_coef
        self._other_coef = other_coef

    def set_lr_scheduler(self,
                         patience=None,
                         factor=None,
                         threshold=None,
                         use_lr_scheduler=None):
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
        exe = BasicExecutor()
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


def get_executor(model, config, executor_weights_path=None):
    if isinstance(config, str):
        config = read_json(config)

    builder = ExecutorBuilder(config['use_native'])
    builder.set_optimizer(**config['optim_config'])
    builder.set_loss(**config['loss_config'])
    builder.set_lr_scheduler(**config['lr_scheduler_config'])
    params = get_params(model, **config['weight_decay_config'])
    executor = builder.build(params,
                             executor_weights_path=executor_weights_path)
    return executor
