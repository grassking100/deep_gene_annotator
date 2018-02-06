"""This submodule provide facade to build model"""
from . import CustomObjectsFacade
from . import SeqAnnModelBuilder
from . import CnnSettingBuilder
from . import AttrValidator
class ModelFacade():
    def __init__(self,setting,weights):
        self._total_convolution_layer_size = setting['total_convolution_layer_size']
        self._convolution_layer_numbers = setting['convolution_layer_numbers']
        self._convolution_layer_sizes = setting['convolution_layer_sizes']
        self._terminal_signal = setting['terminal_signal']
        self._annotation_types = setting['ANN_TYPES']
        self._output_dim = setting['output_dim']
        self._lstm_layer_number = setting['lstm_layer_number']
        self._add_batch_normalize = setting['add_batch_normalize']
        self._dropout = setting['dropout']
        self._learning_rate = setting['learning_rate']
        self._weights = weights
        self._cnn_setting = None
        self._custom_objects = None
        self._build_cnn_setting()
        self._build_custom_objects()
    def _validate_attrs(self,attrs):
        attr_validator = AttrValidator(self)
        attr_validator.validated_keys = attrs
        attr_validator.invalid_values = [None]
        attr_validator.validate()
    def _validate(self):
        """Validate if all attribute is set correctly"""   
        attr_validator = AttrValidator(self)
        attr_validator.is_protected_validated = True
        attr_validator.validate()
    def _build_cnn_setting(self):
        prefix = "_"+self.__class__.__name__+"__"
        self._validate_attrs([prefix+'convolution_layer_numbers',
                              prefix+'convolution_layer_sizes'])
        size = self._total_convolution_layer_size
        if not(size == len(self._convolution_layer_numbers) and size == len(self._convolution_layer_sizes)):
            raise Exception("Size is not consistent")
        cnn_setting_builder=CnnSettingBuilder()
        for i in range(self._total_convolution_layer_size):
            cnn_setting_builder.add_layer(self._convolution_layer_numbers[i],
                                          self._convolution_layer_sizes[i])
        self._cnn_setting = cnn_setting_builder.build()
    def _build_custom_objects(self):
        prefix = "_"+self.__class__.__name__+"__"
        self._validate_attrs([prefix+'annotation_types',
                              prefix+'output_dim',
                              prefix+'terminal_signal'])
        facade = CustomObjectsFacade(self._annotation_types,
                                     self._output_dim,self._terminal_signal,
                                     self._weights,
                                     'accuracy','loss')
        self._custom_objects = facade.custom_objects
    def model(self):
        """Method to build model"""
        self._validate()
        model_builder = SeqAnnModelBuilder()
        model_builder.cnn_setting = self._cnn_setting
        model_builder.lstm_layer_number = self._lstm_layer_number
        model_builder.terminal_signal = self._terminal_signal
        model_builder.add_batch_normalize = self._add_batch_normalize
        model_builder.dropout = self._dropout
        model_builder.learning_rate = self._learning_rate
        model_builder.output_dim = self._output_dim
        model_builder.custom_objects = self._custom_objects
        return model_builder.build()
