from .process import get_time_str
import pandas as pd
import subprocess
import nvgpu

class Process:
    def __init__(self,cmd,name=None):
        self.name=name
        self.cmd=cmd
        self._process = None
        self._returned_code = None
        self._is_start = False
        self._start_time = None
        self._end_time = None
        self._recorded_args = None

    @property
    def start_time(self):
        return self._start_time
        
    @property
    def end_time(self):
        return self._end_time
        
    @property
    def is_start(self):
        return self._is_start
        
    @property
    def returned_code(self):
        return self._returned_code
        
    def start(self,*args):
        if not self.is_start:
            self._start_time = time.gmtime()
            print(self.cmd.format(*args))
            self._process = subprocess.Popen(self.cmd.format(*args),shell=True)
            self._is_start = True
            self._recorded_args = args
        
    @property
    def record(self):
        record = {'is start':self.is_start,
                  'is finish':self.is_finish,
                  'start time':get_time_str(self.start_time),
                  'end time':get_time_str(self.end_time),
                  'name':self.name,
                  'cmd':self.cmd,
                  'returned code':self.returned_code,
                  'args':self._recorded_args}
        return record

    @property
    def is_finish(self):
        if self.is_start:
            if self._returned_code is not None:
                return True
            else:
                self._returned_code = self._process.poll()
                is_finish = self._returned_code is not None
                if is_finish:
                    self._end_time = time.gmtime()
                return is_finish
        else:
            return False
        
def process_schedule(processes,gpu_ids,mem_used_percent_threshold=None):
    mem_used_percent_threshold = mem_used_percent_threshold or 1
    processes = dict(zip(list(range(len(processes))),processes))
    gpu_ready = [None] * len(gpu_ids)
    while True:
        for index,belong_id in enumerate(gpu_ready):
            if belong_id is None:
                gpus = nvgpu.gpu_info()
                gpu = list(filter(lambda x:x['index']==str(gpu_ids[index]),gpus))
                ready_ps = [p for p in processes.values() if not p.is_start]
                if len(ready_ps)>0 and len(gpu)==1:
                    if gpu[0]['mem_used_percent']<=mem_used_percent_threshold:
                        ready_p = ready_ps[0]
                        ready_p.start(gpu_ids[index])
                        gpu_ready[index]=ready_p.name
            else:
                p = processes[belong_id]
                if p.is_finish:
                    gpu_ready[index] = None

        if all([p.is_finish for p in processes.values()]):
            break
    return pd.DataFrame.from_dict([p.record for p in processes.values()])
