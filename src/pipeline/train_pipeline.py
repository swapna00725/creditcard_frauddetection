#packages import
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd
from src.components import CustomException
from src.components import logging


if __name__=="__main__":
    try:
        logging.info("training pipeline has started")
        ingest=DataIngestion()
        train_data,test_data=ingest.initiate_data_ingestion()
        transf=DataTransformation()
        train_arr,test_arr=transf.initiate_data_transformation(train_data,test_data)
        trainer=ModelTrainer()
        trainer.initiate_model_trainer(train_arr,test_arr)
        logging.info('pipeline completed')
    except Exception as e:
        logging.error("error has occurred while train pipeline is happening")
        raise CustomException(e,sys)


        
       




