"""This submodule provides class to defined Pipeline"""
from abc import ABCMeta
from time import strftime, gmtime, time
from ..utils.utils import create_folder

class Pipeline(metaclass=ABCMeta):
    def __init__(self,worker,path=None):
        self.worker = worker
        self.path = path
        self.is_prompt_visible = True

    def print_prompt(self,value):
        if self.is_prompt_visible:
            print(value)

    def execute(self):
        if self.path is not None:
            self.print_prompt("Creating folder "+self.path+"...")
            create_folder(self.path)
        self.print_prompt("Processing worker...")
        self._before_execute()
        self.print_prompt("Executing...")
        self._execute()
        self._after_execute()

    def _before_execute(self):
        self.worker.before_work(path=self.path)

    def _after_execute(self):
        self.worker.after_work(path=self.path)

    def _execute(self):
        if self.is_prompt_visible:
            print('Start working('+strftime("%Y-%m-%d %H:%M:%S",gmtime())+")")
        start_time = time()
        self.worker.work()
        end_time = time()
        time_spend = end_time - start_time
        if self.is_prompt_visible:
            print('End working(' + strftime("%Y-%m-%d %H:%M:%S",gmtime()) + ")")
            print("Spend time: " + strftime("%H:%M:%S", gmtime(time_spend)))
