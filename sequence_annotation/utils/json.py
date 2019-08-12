"""This submodule provides class to parse setting file"""
from pathlib import Path
import json

def read_json(path):
    """Read file and parse to json format"""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    with open(path) as file:    
        setting = json.load(file) 
    return setting
