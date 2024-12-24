import os
import sys
import logging
import pickle

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("TextClassifierLogger")


import pickle

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def save_to_dataframe(**kwargs):
    """
    Convert a flexible number of single-valued variables into a DataFrame with their names and values.
    
    Parameters:
    - **kwargs: Variable number of keyword arguments representing variable names and their single values.

    Returns:
    - DataFrame with 'Name' and 'Value' columns.
    """
    # Create a DataFrame directly from the kwargs
    pd.DataFrame({'Name': list(kwargs.keys()), 'Value': list(kwargs.values())}).to_csv(os.path.join('artifacts',"trained_model_metrics.csv"))