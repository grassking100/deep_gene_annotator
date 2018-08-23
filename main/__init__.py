'''This modeule proviodes pipeline of traning,testing model'''
import os
import sys
sys.path.append(os.path.abspath(os.path.expanduser(__file__+"/../..")))
from sequence_annotation.pipeline.train_pipeline import TrainPipeline
from sequence_annotation.pipeline.worker.train_worker import TrainWorker
