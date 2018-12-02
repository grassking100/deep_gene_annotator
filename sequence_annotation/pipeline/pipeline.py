"""This submodule provides class to defined Pipeline"""
from abc import ABCMeta
from time import strftime, gmtime, time

class Pipeline(metaclass=ABCMeta):
    def __init__(self,model_processor,data_processor,compiler,worker,wrapper,is_prompt_visible=True,id_=None,path=None):
        self._id = id_
        self._path = path
        if self._path is not None and self._id is not None:
            self._path = self._path + "/" + str(self._id)
        self._is_prompt_visible = is_prompt_visible
        self._worker = worker
        self._compiler = compiler
        self._data_processor = data_processor
        self._model_processor = model_processor
        self._wrapper = wrapper
    def print_prompt(self,value):
        if self._is_prompt_visible:
            print(value)
    def execute(self):
        self.print_prompt("Processing model..")
        self._prepare_model()
        self.print_prompt("Processing data...")
        self._prepare_data()
        self.print_prompt("Compiling model...")
        self._compile_model()
        self.print_prompt("Processing worker...")
        self._prepare_worker()
        self._before_execute()
        self.print_prompt("Executing...")
        self._execute()
        self._after_execute()
    def _compile_model(self):
        self._compiler.before_process(self._path)
        self._compiler.process(self._model_processor.model)
        self._compiler.after_process(self._path)
    def _prepare_model(self):
        self._model_processor.before_process(self._path)
        self._model_processor.process()
        self._model_processor.after_process(self._path)
    def _prepare_data(self):
        self._data_processor.before_process(self._path)
        self._data_processor.process()
        self._data_processor.after_process(self._path)
    def _prepare_worker(self):
        self._worker.model=self._model_processor.model
        self._worker.data=self._data_processor.data
        self._worker.wrapper = self._wrapper
    def _before_execute(self):
        self._worker.before_work()
    def _after_execute(self):
        self._worker.after_work()
    def _execute(self):
        if self._is_prompt_visible:
            print('Start working('+strftime("%Y-%m-%d %H:%M:%S",gmtime())+")")
        start_time = time()
        self._worker.work()
        end_time = time()
        time_spend = end_time - start_time
        if self._is_prompt_visible:
            print('End working(' + strftime("%Y-%m-%d %H:%M:%S",gmtime()) + ")")
            print("Spend time: " + strftime("%H:%M:%S", gmtime(time_spend)))