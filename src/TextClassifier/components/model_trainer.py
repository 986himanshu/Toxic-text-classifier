import os
import sys
from dataclasses import dataclass
from TextClassifier.utils import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional, Dense
from TextClassifier.utils import *

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    defined_model_summary_file_path=os.path.join("artifacts","model.png")

class ModelTrainer:
    def __init__(self, train, test, validation):
        self.config=ModelTrainerConfig()
        self.train=train
        self.test=test
        self.validation=validation
        self.model = Sequential(name="text-classifier")
        self.model.add(Embedding (len(tokenizer.get_vocab()), 32))
        self.model.add(Bidirectional (LSTM (32, activation='tanh')))
        self.model.add(Dense (128, activation='relu'))
        self.model.add(Dense (256, activation='relu'))
        self.model.add(Dense (128, activation='relu'))
        self.model.add(Dense (1, activation='sigmoid'))

        self.model.summary()

    def build_model(self):
        logging.info("Defining model & compiling model")



        # Write model summary in png

        logger.info('Writing model summary')

        logger.info('Compliling model with crossentropy loss function and adam optimization')

        self.model.compile(loss="binary_crossentropy", optimizer='Adam')


        # Writting some code for performance enhancement using popular numbers in documentation
        self.train = self.train.cache()
        self.train = self.train.shuffle(buffer_size=16000)
        self.train = self.train.batch(16) #To look for reason why interviewer might ask to use
        self.train = self.train.prefetch(buffer_size= 8
                                #tf.data.experimental.AUTOTUNE suggested by autocomplete
                                )
        history = self.model.fit(self.train, epochs=1, batch_size=16, validation_data=self.validation)
        save_object(
                file_path=self.ModelTrainerConfig.trained_model_path,
                obj=self.model
            )
        

        
