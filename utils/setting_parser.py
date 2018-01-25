"""This submodule provides class to parse setting file"""
from abc import ABCMeta, abstractmethod
import configparser
def str2bool(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise Exception((str)(value)+" is neithor True of False")
class SettingParser(metaclass=ABCMeta):
    """The class provide method to parse ini format file"""
    def __init__(self, setting_file):
        self._config = configparser.ConfigParser()
        self._config.read(setting_file)
        self._setting = None
    @abstractmethod
    def parse(self):
        """Parse setting"""
        pass
    @property
    def setting(self):
        """Get setting"""
        if self._setting is None:
            raise Exception("There is not setting yet")
        return self._setting
    def _add_values(self, setting_, keys, type_convert_function):
        """Convert values into specific type and add to setting"""
        temp_setting = {}
        for key in keys:
            temp_setting[key] = type_convert_function(setting_[key])
        self._setting.update(temp_setting)
    def _add_vector_of_string_values(self, setting_, keys):
        """Convert values into vectors of string and add to setting"""
        self._add_vectors_of_value(setting_, keys, str)
    def _add_vectors_of_value(self, setting_, keys, type_convert_function):
        """Convert vectors of value into specific type and add to setting"""
        temp_setting = {}
        for key in keys:
            temp_setting[key] = [type_convert_function(i) for i in setting_[key].split(",")]
        self._setting.update(temp_setting)
    def _add_bool_values(self, setting_, keys):
        """Convert values into boolean and add to setting"""
        self._add_values(setting_, keys, str2bool)
    def _add_str_values(self, setting_, keys):
        """Convert values into string type and add to setting"""
        self._add_values(setting_, keys, str)
    def _add_int_values(self, setting_, keys):
        """Convert values into int type and add to setting"""
        self._add_values(setting_, keys, int)
    def _add_float_values(self, setting_, keys):
        """Convert values into float type and add to setting"""
        self._add_values(setting_, keys, float)
    def _add_vector_of_int_values(self, setting_, keys):
        """Convert values into vectors of int and add to setting"""
        self._add_vectors_of_value(setting_, keys, int)
class ModelSettingParser(SettingParser):
    """The class provide method to parse model setting file"""
    def parse(self):
        """Parse setting"""
        self._setting = {}
        root = "model_setting"
        setting_ = self._config[root]
        key_int_value = ['total_convolution_layer_size', 'lstm_layer_number',
                         'output_dim']
        key_float_value = ['dropout', 'learning_rate']
        key_bool_value = ['add_batch_normalize']
        key_ints_value = ['convolution_layer_sizes',
                          'convolution_layer_numbers']
        key_str_array_value = ['ANN_TYPES']
        self._add_vector_of_string_values(setting_, key_str_array_value)
        self._add_int_values(setting_, key_int_value)
        self._add_float_values(setting_, key_float_value)
        self._add_bool_values(setting_, key_bool_value)
        self._add_vector_of_int_values(setting_, key_ints_value)
class TrainSettingParser(SettingParser):
    """The class provide method to parse training setting file"""
    def __init__(self, setting_file):
        super().__init__(setting_file)
    def __parse_show_settings(self):
        """Parse setting of showing"""
        root = "show"
        setting_ = self._config[root]
        key_bool_value = ['is_model_visible', 'is_verbose_visible', 'is_prompt_visible']
        self._add_bool_values(setting_, key_bool_value)
    def parse(self):
        """Parse setting"""
        self._setting = {}
        self.__parse_training_settings()
        self.__parse_show_settings()
    def __parse_training_settings(self):
        """Parse setting of training"""
        root = "training_settings"
        setting_ = self._config[root]
        key_bool_value = ['use_weights']
        key_int_value = ['step', 'progress_target', 'previous_epoch', 'batch_size']
        key_str_array_value = ['training_files', 'validation_files']
        key_str_value = ['outputfile_root','mode_id',
                         'training_answers',
                         'validation_answers',
                         'previous_status_root']
        self._add_str_values(setting_,key_str_value)
        self._add_int_values(setting_, key_int_value)
        self._add_bool_values(setting_, key_bool_value)
        self._add_vector_of_string_values(setting_, key_str_array_value)
        terminal_signal = setting_['terminal_signal']
        if str(terminal_signal) == "None":
            self._setting['terminal_signal'] = None
        else:
            
            self._setting['terminal_signal'] = int(setting_['terminal_signal'])
