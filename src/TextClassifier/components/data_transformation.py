import os
from textSummarizer.logging import logger
from transformers import BertTokenizer
# from textSummarizer.entity import DataTransformationConfig

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    model_name = 'bert-base-cased'
    preprocessed_data_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name, config.del_name, configdel_name, config.del_name, config.del_name)


    
    def convert_text_to_tensors(self,example_batch):
        X = example_batch['tweet']
        sequences = [sequence for sequence in X]
        model_inputs = tokenizer(sequences,
                         padding=True, #To create inputs equilength (Trasnformation)
                         return_tensors='tf')
        dataset = tf.data.Dataset.from_tensor_slices((model_inputs['input_ids'],y))
        return dataset
    

    def initiate_data_transformation(self,train_path,test_path):
        logger.info('Data transformation started')
        train_dataset = pd.read_csv(train_path)
        test_dataset = pd.read_csv(test_path)
        
        logger.info('Loaded training dataset and test dataset')
        logger.info('Preprocessing datasets and further splitting test datasets into test and validation datasets")
        train = self.convert_text_to_tensors(train_dataset)
        test_val = self.convert_text_to_tensors(test_dataset)
        test = test.take(int(len(test_val) * 0.7))
        val = test.skip(int(len(test_val) * 0.7)).take(int(len(test_val) * 0.3))

        return (train, test, val)


        
        
