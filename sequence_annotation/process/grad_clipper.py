from torch import nn

class GradClipper:
    def __init__(self,clip_grad_value=None,clip_grad_norm=None,grad_norm_type=None):
        self._clip_grad_value = clip_grad_value
        self._clip_grad_norm = clip_grad_norm
        self._grad_norm_type = grad_norm_type or 2
        
    def clip(self,parameters):
        if self._clip_grad_value is not None:
            nn.utils.clip_grad_value_(parameters, self._clip_grad_value)
        elif self._clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(parameters, self._clip_grad_norm,
                                     self._grad_norm_type)
            
    def get_config(self):
        config = {}
        config['class'] = self.__class__.__name__
        config['clip_grad_value'] = self._clip_grad_value
        config['clip_grad_norm'] = self._clip_grad_norm
        config['grad_norm_type'] = self._grad_norm_type
        return config
