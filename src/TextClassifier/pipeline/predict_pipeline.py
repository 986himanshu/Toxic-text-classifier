import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from transformers import BertTokenizer


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,text):
        try:
            model_path='artifacts\model.pkl'
            model=load_object(file_path=model_path)
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            model_input = tokenizer(text,
                            padding='max_length', #To create inputs equilength (Transformation)
                            return_tensors='tf')            
            preds=model.predict(model_input['input_ids'].numpy())
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

