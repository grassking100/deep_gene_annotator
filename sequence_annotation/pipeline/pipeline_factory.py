from abc import ABCMeta, abstractmethod
from . import TrainPipeline,TestPipeline
from . import SeqAnnDataHandler,SimpleDataHandler
class PipelineFactory(metaclass=ABCMeta):
    def create(self,purpose,data_type="simple",is_prompt_visible=True):
        if purpose == 'train':
            pipeline = TrainPipeline(is_prompt_visible)
        elif purpose == 'test':
            pipeline = TestPipeline(is_prompt_visible)
        else:
            raise Exception("Purpose,"+purpose+", is not supported")
        if data_type == 'simple':
            pipeline.data_handler = SimpleDataHandler
        elif data_type == 'sequence_annotation':
            pipeline.data_handler = SeqAnnDataHandler
        else:
            raise Exception("Data type,"+data_type+", is not supported")
        return pipeline