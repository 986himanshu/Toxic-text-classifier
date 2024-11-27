import os
from pathlib import Path
import logging

logging.basicConfig(level= logging.INFO, format= '[%(asctime)s]: %(message)s:')

project_name = "TextClassifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/init.py",
    f"src/{project_name}/components/init.py",
    f"src/{project_name}/utils/init.py",
    f"src/{project_name}/config/init.py",
    f"src/{project_name}/config/configuration/init.py",
    f"src/{project_name}/pipeline/init.py",
    f"src/{project_name}/entity/init.py",
    "config/config.yaml",
    "dvc.yaml",
    "requirements.txt",
    "setup.py",
    "research/live_code.ipynb",
    "templates/index.html"

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")
    if (not os.path.exists(filepath) or (os.path.getsize(filepath)== 0)):
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")
    else:
        logging.info(f"{filename} already exists")
