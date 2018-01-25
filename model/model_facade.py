"""This submodule provide facade to build model"""
from . import CustomObjectsFacade
from . import SeqAnnModelBuilder
from . import CnnSettingBuilder

class ModelFacade():
    def __init__(self,setting):
        self.__total_convolution_layer_size = setting['total_convolution_layer_size']
        self.__convolution_layer_numbers = setting['convolution_layer_numbers']
        self.__convolution_layer_sizes = setting['convolution_layer_sizes']
        self.__terminal_signal = setting['terminal_signal']
        self.__annotation_types = setting['ANN_TYPES']
        self.__output_dim = setting['output_dim']
        self.__lstm_layer_number = setting['lstm_layer_number']
        self.__add_batch_normalize = setting['add_batch_normalize']
        self.__dropout = setting['dropout']
        self.__learning_rate = setting['learning_rate']
        self.__cnn_setting = None
        self.__custom_objects = None
        self._build_cnn_setting()
        self._build_custom_objects()
    def _validate(self,attrs):
        """Validate if all attribute is set correctly"""
        for attr in attrs:
            if getattr(self,attr) is None:
                raise Exception("Facade needs "+attr+" to complete the quest")      
                
    def _build_cnn_setting(self):
        prefix = "_"+self.__class__.__name__+"__"
        self._validate([prefix+'convolution_layer_numbers',
                        prefix+'convolution_layer_sizes'])
        size = self.__total_convolution_layer_size
        if not(size == len(self.__convolution_layer_numbers) and size == len(self.__convolution_layer_sizes)):
            raise Exception("Size is not consistent")
        cnn_setting_builder=CnnSettingBuilder()
        for i in range(self.__total_convolution_layer_size):
            cnn_setting_builder.add_layer(self.__convolution_layer_numbers[i],
                                          self.__convolution_layer_sizes[i])
        self.__cnn_setting = cnn_setting_builder.build()
    def _build_custom_objects(self):
        prefix = "_"+self.__class__.__name__+"__"
        self._validate([prefix+'annotation_types',
                        prefix+'output_dim',
                        prefix+'terminal_signal'])
        facade = CustomObjectsFacade(self.__annotation_types,
                                     self.__output_dim,self.__terminal_signal,
                                     'accuracy','loss')
        self.__custom_objects = facade.custom_objects
    def model(self):
        """Method to build model"""
        prefix = "_"+self.__class__.__name__+"__"
        attrs = [attr for attr in dir(self) if attr.startswith(prefix)]
        self._validate(attrs)
        model_builder = SeqAnnModelBuilder()
        model_builder.cnn_setting = self.__cnn_setting
        model_builder.lstm_layer_number = self.__lstm_layer_number
        model_builder.terminal_signal = self.__terminal_signal
        model_builder.add_batch_normalize = self.__add_batch_normalize
        model_builder.dropout = self.__dropout
        model_builder.learning_rate = self.__learning_rate
        model_builder.output_dim = self.__output_dim
        model_builder.custom_objects = self.__custom_objects
        return model_builder.build()
