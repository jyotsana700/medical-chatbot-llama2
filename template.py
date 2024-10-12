import os
import logging
from pathlib import Path

list_of_files=[
    "src/__init__.py",
    "src/helper.py",
    "env",
    "research/trials.py",
    "app.py",
    "store_index.py",
    "static",
    "templates/chat.html"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir,exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filename)):
        with open(filepath,'w') as f:
            pass
    else:
        pass