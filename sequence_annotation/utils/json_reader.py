"""This submodule provides class to parse setting file"""
from pathlib import Path
import os
import json
def str2bool(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise Exception((str)(value)+" is neithor True of False")
class JsonReader:
    """The class provide method to parse json format file"""
    def _validate_file_exist(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    def read(self,path):
        """Read file and parse to json format"""
        setting = ""
        self._validate_file_exist(path)
        with open(path) as file:    
            setting = json.load(file) 
        return setting
