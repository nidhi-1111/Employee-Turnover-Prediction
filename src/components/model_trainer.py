import os
import sys
from dataclasses import dataclass
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path_file = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splittting train/test data')
            x_train, y_train, x_test, y_test = (train_array[:,:-1], train_array[:,-1], 
                                                test_array[:,:-1], test_array[:,-1])
            
            models = {
                    "Random Forest" : RandomForestClassifier(),
                    "Logistic Regression" : LogisticRegression(max_iter=2000)

            }

            models_report:dict = evaluate_models(x_train= x_train, y_train=y_train, x_test=x_test, y_test=y_test, models = models)
        
            # To get the best score/model

            best_model_score = max(sorted(models_report.values()))
            best_model_name = list(models_report.keys())[list(models_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found")

            save_object(
                file_path=self.model_trainer_config.trained_model_path_file,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            acc = accuracy_score(y_test,predicted)

            return acc
        
        except Exception as e:
            raise CustomException(e,sys)

