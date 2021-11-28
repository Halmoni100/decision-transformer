import os
import shutil
import pathlib

def rm_and_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)