import os
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components import CustomException
from src.components import logging


import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw.csv')
class DataIngestion:
      def __init__(self):
           self.ing_config=DataIngestionConfig()
      def initiate_data_ingestion(self):
           try:
                logging.info("data ingestion has started")
                df=pd.DataFrame('notebook/data.scv')
                os.makedirs(os.path.join(self.ing_config.train_data_path),exists_ok=True)
                df.to_csv('self.ing_config.raw_data_path',index=False,header=True)
                logging.info("train and test dataset are assigned")
                train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
                train_set.to_csv('self.ing_config.train_data_path',index=False,header=True)
                test_set.to_csv('self.ing_config.test_data_path',index=False,header=True)
                logging.info('train and test data paths are set')
                 
                return(self.ing_config.train_data_path,self.ing_config.test_data_path) 
           except Exception as e:
                logging.error("data ingsetion failed")
                raise CustomException(e,sys)
    