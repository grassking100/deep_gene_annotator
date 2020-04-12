import pandas as pd
import subprocess
import nvgpu
from .utils import get_time_str


class Process:
    def __init__(self, cmd, name=None, no_gpu=False):
        self.name = name
        self._cmd = cmd
        if not no_gpu:
            self._cmd += " -g {}"
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

    def start(self, *args):
        if not self.is_start:
            self._start_time = get_time_str()
            print(self._cmd.format(*args))
            self._process = subprocess.Popen(self._cmd.format(*args),
                                             shell=True)
            self._is_start = True
            self._recorded_args = args

    @property
    def record(self):
        record = {
            'is start': self.is_start,
            'is finish': self.is_finish,
            'start time': self.start_time,
            'end time': self.end_time,
            'name': self.name,
            'cmd': self._cmd,
            'returned code': self.returned_code,
            'args': self._recorded_args
        }
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
                    self._end_time = get_time_str()
                return is_finish
        else:
            return False


def process_schedule(processes,
                     gpu_ids,
                     mem_used_percent_threshold=None,
                     no_gpu=False):
    if no_gpu:
        process_unit_ids = list(range(40))
    else:
        mem_used_percent_threshold = mem_used_percent_threshold or 1
        if len(set(gpu_ids)) != len(gpu_ids):
            raise Exception("Duplicated gpu id")
        process_unit_ids = gpu_ids
    processes_ = processes
    processes = {}
    for index, p in enumerate(processes_):
        p.name = index
        processes[index] = p
    resource_ready = [None] * len(process_unit_ids)
    while True:
        for index, belong_id in enumerate(resource_ready):
            if belong_id is None:
                ready_ps = [p for p in processes.values() if not p.is_start]
                if len(ready_ps) > 0:
                    if no_gpu:
                        ready_p = ready_ps[0]
                        ready_p.start(process_unit_ids[index])
                        resource_ready[index] = ready_p.name
                    else:
                        gpus = nvgpu.gpu_info()
                        gpu = list(
                            filter(
                                lambda x: x['index'] == str(process_unit_ids[
                                    index]), gpus))
                        if gpu[0][
                                'mem_used_percent'] <= mem_used_percent_threshold:
                            ready_p = ready_ps[0]
                            ready_p.start(process_unit_ids[index])
                            resource_ready[index] = ready_p.name
            else:
                p = processes[belong_id]
                if p.is_finish:
                    resource_ready[index] = None

        if all([p.is_finish for p in processes.values()]):
            break
    return pd.DataFrame.from_dict([p.record for p in processes.values()])
