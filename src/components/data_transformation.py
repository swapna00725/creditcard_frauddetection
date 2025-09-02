import os
import sys

import pandas as pd
from src.components import CustomException
from src.components import logging

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from dataclasses import dataclass
@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.transf_config=DataTransformationConfig()

    def get_transformer_object(self):
        try:
            logging.info("separetae numerical cols and cat cols")
            num_cols=[' ',' ', ' ', ' ']
            cat_cols=[' ', ' ', ' ', ' ']
            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer()),('scaler',StandardScaler())])
            cat_pipeline=Pipeline(steps=[('imputer',SimpleImputer()),('scaler',StandardScaler()),('encoder',OneHotEncoder())])
            preprocessor=ColumnTransformer([('numpipe',num_pipeline,num_cols),('catpipe',cat_pipeline,cat_cols)])
            logging.info("preprocessor process completed")

            return preprocessor

        except Exception as e:
            CustomException(e,sys)

        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("making use of preprocessor object")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            preprocessor_obj=self.get_transformer_object()

            target=['target']
            X_train_df=train_df.drop(columns=['target'],axis=1)
            y_train_df=train_df.drop(columns=['target'],axis=1)
            X_test_df=test_df.drop(columns=['target'],axis=1)
            y_test_df=test_df.drop(columns=['target'],axis=1)
        
            X_train_arr=preprocessor_obj.fit_transform(X_train_df)
            X_test_arr=preprocessor_obj.transform(X_train_df)

            train_arr=np.c_(X_train_arr,np.array(y_train_df))
            test_arr=np.c_(X_test,np.array(y_test_df))

            save_object(file_path=self.transf_config.preprocessor_obj_file_path,obj=preprocessor_obj)

            return(train_arr,test_arr,self.transf_config.preprocessor_obj_file_path)
        except Exception as e:
            raise CustomException(e,sys)


 