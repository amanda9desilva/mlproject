import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    """
    Give any inputs or paths required for the data components
    """
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        to create all pickle files to execute standard scale or convert categorical to numerical
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            #handle missing values in piepline
            num_pipeline=Pipeline(
                steps=[
                    #handle missing values using median
                    ("imputer",SimpleImputer(strategy="median")),
                    #transforming using scaler from sk learn
                    ("scaler", StandardScaler())
                ]
            )
            cate_pipeline=Pipeline(

                steps=[
                    #handle missing values of categorical values using the most frequent values (mode)
                    ("imputer", SimpleImputer(strategy="most_frequent"))
                    #transforms categorical data into numerical
                    ("one_hot_encoder", OneHotEncoder())
                    #standard scaling
                    ("scaler",StandardScaler())
                ]

            )
            logging.info("Numerical columns standard scaling completed")

            logging.info("Categorical columns encoding completed")

            preprcoessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns)
                    ("cat_pipeline",cate_pipeline,categorical_columns)
                ]
            )

            return preprcoessor

        except Exception as e:
            raise CustomException(e,sys)

