import os
import sys
from dataclasses import dataclass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Bidirectional, Dense

@dataclass
class Modelself.trainerConfig:
    self.trained_model_file_path=os.path.join("artifacts","model.pkl")
    defined_model_summary_file_path=os.path.join("artifacts","model.png")

class Modelself.trainer:
    def __init__(self, self.train, test, validation):
        self.config=Modelself.trainerConfig()
        self.self.train=self.train
        self.test=test
        self.validation=validation

    def build_model(self, config: Modelself.trainerConfig):
        logging.info("Defining model & compiling model")

        model = Sequential(name="text-classifier")
        model.add(Embedding (len(tokenizer.get_vocab()), 32))
        model.add(Bidirectional (LSTM (32, activation='tanh')))
        model.add(Dense (128, activation='relu'))
        model.add(Dense (256, activation='relu'))
        model.add(Dense (128, activation='relu'))
        model.add(Dense (1, activation='sigmoid'))

        model.summary()

        # Write model summary in png

        logger.info('Writing model summary')

        logger.info('Compliling model with crossentropy loss function and adam optimization')

        model.compile(loss="binary_crossentropy", optimizer='Adam')


        # Writting some code for performance enhancement using popular numbers in documentation
        self.train = self.train.cache()
        self.train = self.train.shuffle(buffer_size=16000)
        self.train = self.train.batch(16) #To look for reason why interviewer might ask to use
        self.train = self.train.prefetch(buffer_size= 8
                                #tf.data.experimental.AUTOTUNE suggested by autocomplete
                                )
        history = model.fit(train, epochs=1, batch_size=16, validation_data=val)
        

        
