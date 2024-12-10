import os
import sys
from TextClassifier.exception import CustomException
from TextClassifier.logger import *
from TextClassifier.components.model_trainer import ModelTrainer
from TextClassifier.components.data_transformation import DataTransformation
from TextClassifier.utils import *
import pandas as pd
from TextClassifier.components.model_evaluation import ModelEvaluator
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# from TextClassifier.components.data_transformation import DataTransformation

# from TextClassifier.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('research/data/tweets.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train,test,val=data_transformation.initiate_data_transformation(obj.ingestion_config.train_data_path,obj.ingestion_config.test_data_path)

    model_trained = modeltrainer=ModelTrainer(train,test,val)
    model_trained.buildModel()

    model_eval = ModelEvaluator(model_trained.ModelTrainerConfig)

    (precision, recall, accuracy) = model_eval.evaluate_model()

    save_to_csv(precision, recall, accuracy)






