import os
import sys

import pandas as pd
from src.components import CustomException
from src.components import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


from dataclasses import dataclass
@dataclass 
class ModelTrainerConfig:
    model_pkl_file_path=os.path.join('artifacts','credit_model.pkl')
class ModelTrainer:
    def __init__(self):
        self.trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("model building started")

            Xtr,ytr,Xte,yte=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]
            models={'RandomForest': 'RandomForestClassifier()',
                    'LogisticRegression': 'LogisticRegression()',
                    'DecisionTree': 'DecisionTreeClassifier()',
                    'AdaBoostClassifier': 'AdaBoostClassifier()',
                    'GradientBoost': 'GradientBoostingClassifier()',
                    'Kneignhbors': 'KNeighborsClassifier()'
                    }
       
            params={'RandomForest':[n_estimators:[4,8,12],max_depth=[4,6,8,10],max_leaves=[1,10]]
                    'decision tree':{max_depth:[1,2,3],.......................
                    }}

            model_report : dict = evaluate_models(Xtr,ytr,Xte,yte,models,params)

            best_model_score=max(list(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            logging.info("the best model is {best_model}")
            print(f"the best model is {best_model}")
             
            save_object(file_path=self.trainer_config.model_pkl_file_path,obj='best_model')
            pred=best_model.predict(Xte)
            acc=accuracy_score(yte,pred)

            logging.info("the prediction from the best model is {pred}")
            print(f"the prediction from the best model is {pred}")

            return acc
        except Exception as e:
                
            raise CustomException(e,sys)