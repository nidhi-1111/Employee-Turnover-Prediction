import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        Renames and transforms categorical feature into numerical
        
        """
        try:
            cat_features = ['sales','salary']

            cat_pipeline = Pipeline(
                steps=[
                       ('label_encoder', OneHotEncoder())
                ]
            )
            logging.info('Categorical encoding and renaming completed')

            preprocessor = ColumnTransformer(
                [('cat_pipelines', cat_pipeline, cat_features)],
                remainder='passthrough' 
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f'Read the train and test completed')
            logging.info(f'obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'left'

            # cat_features = ['sales','salary']
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f'Applying preprocessor obj on training and tetsing dataframe')

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df) 
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df) 

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path =  self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)  
           
