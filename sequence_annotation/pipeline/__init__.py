'''This modeule proviodes pipeline of traning,testing model'''
from ..model.custom_objects import CustomObjectsFacade
from ..model.model_trainer import ModelTrainer
from ..data_handler.training_data_handler import handle_alignment_files
from ..model.model_build_facade import ModelBuildFacade
from ..utils.setting_parser import TrainSettingParser, ModelSettingParser
from ..utils.data_loader import TrainDataLoader
from .pipeline import Pipeline