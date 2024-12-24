from TextClassifier.utils import *
from TextClassifier.components.model_trainer import ModelTrainerConfig
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

class ModelEvaluator:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.model = load_object(self.config.model_path)

    def evaluate_model(self):
        pre = Precision()
        rec = Recall()
        acc = BinaryAccuracy()
        for batch in test_data.as_numpy_iterator():
            x_true, y_true = batch
            y_hat = self.model.predict(x_true)
            if y_true.shape != y_hat.shape:
                y_true = y_true.reshape(-1)
                y_hat = y_hat.reshape(-1)
            pre.update_state(y_true, y_hat)
            rec.update_state(y_true, y_hat)
            acc.update_state(y_true, y_hat)
        return pre.result(), rec.result(), acc.result()
