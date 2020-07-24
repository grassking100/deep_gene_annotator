import torch
import pandas as pd
import subprocess
import nvgpu
from .utils import get_time_str


class Process:
    def __init__(self, cmd, name=None):
        self.name = name
        self._cmd = cmd
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

    def start(self, gpu_id=None):
        if not self.is_start:
            self._start_time = get_time_str()
            if gpu_id is not None:
                self._cmd += " -g {}"
                self._cmd = self._cmd.format(*gpu_id)
                self._recorded_args = gpu_id
            print(self._cmd)
            self._process = subprocess.Popen(self._cmd,shell=True)
            self._is_start = True
            

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


def process_schedule(processes_list,resource_ids=None,
                     mem_used_percent_threshold=None,
                     use_gpu=True):
    if not use_gpu:
        resource_ids = resource_ids or list(range(40))
    else:
        resource_ids = resource_ids or list(range(torch.cuda.device_count()))
        mem_used_percent_threshold = mem_used_percent_threshold or 5
        if len(set(resource_ids)) != len(resource_ids):
            raise Exception("Duplicated gpu id")
    print("Use resources {} to run {} processes".format(resource_ids,len(processes_list)))
    processes = {}
    for index, p in enumerate(processes_list):
        p.name = index
        processes[index] = p
    resource_status = {}
    for id_ in resource_ids:
        resource_status[str(id_)] = None
    while True:
        for resource_id, process_id in resource_status.items():
            if process_id is None:
                unstart_processes = [p for p in processes.values() if not p.is_start]
                if len(unstart_processes) > 0:
                    has_resource = False
                    if not use_gpu:
                        has_resource = True
                    else:
                        gpus = nvgpu.gpu_info()
                        gpu = list(filter(lambda x: x['index'] == resource_id, gpus))
                        mem_used_percent = gpu[0]['mem_used_percent']
                        if mem_used_percent <= mem_used_percent_threshold:
                            print("GPU {} has {} of memory be used".format(resource_id,mem_used_percent))
                            has_resource = True
                    if has_resource:
                        process = unstart_processes[0]
                        if use_gpu:
                            process.start(gpu_id=resource_id)
                        else:
                            process.start()
                        resource_status[resource_id] = process.name
            else:
                process = processes[process_id]
                if process.is_finish:
                    resource_status[resource_id] = None

        if all([p.is_finish for p in processes.values()]):
            break
    df = pd.DataFrame.from_dict([p.record for p in processes.values()])
    return df
