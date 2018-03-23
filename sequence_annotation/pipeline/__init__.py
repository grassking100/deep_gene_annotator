'''This modeule proviodes pipeline of traning,testing model'''
from ..model import CustomObjectsFacade
from ..model import ModelHandler
from ..worker import ModelTrainer
from ..utils.setting_parser import SettingParser
from ..data_handler import SeqAnnDataHandler,SimpleDataHandler
from .pipeline import Pipeline