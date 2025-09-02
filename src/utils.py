import os
import sys

import pandas as pd
from src.components import CustomException
from src.components import logging

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirnme(file_path,exists_ok=True))
        with open('file_path','wb') as f:
            pickle.dump(obj,f)
    except Exception as e:
            raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open('file_path','rb') as f:
            return pickle.load(f)
    except Exception as e:
            raise CustomException(e,sys)
    
def evaluate_models(Xtr,ytr,Xte,yte,models,params):
     try:
          report={}
          for i in range(len(models.keys())):
               model=list(models.values())[i]
               para=params[list(models.keys())[i]]

               gs=GridSearchCV(model,para,cv=3)
               gs.fit(Xtr,ytr)

               model.set_param(**gs.best_params_)

               tr_pred=model.predict(Xtr)
               te_pred=model.predict(Xte)

               tr_acc=accuracy_score(ytr,tr_pred)
               te_acc=accuracy_score(yte,te_pred)

               report[list(models.keys())[i]]=te_acc

               return report
     except Exception as e:
          raise CustomException(e,sys)      
    
