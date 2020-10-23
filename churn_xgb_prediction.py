# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 02:49:52 2020

@author: Diane Tuyizere
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import auc
import pickle
 
data=pd.read_csv('https://raw.githubusercontent.com/Diane10/ML/master/Customer-Churn.csv')
print(data.head())

data.Churn.value_counts()
categorical_feature_columns = list(set(data.columns) - set(data._get_numeric_data().columns))
categorical_feature_columns
newdata= data.drop(['TotalCharges'],axis=1)
cols=['tenure','Contract','OnlineSecurity','TechSupport','OnlineBackup','DeviceProtection','MonthlyCharges','Churn']
newdata=data[cols]
le=LabelEncoder()
 
for column in newdata.columns:
    if newdata[column].dtype==np.number:
      continue
    newdata[column]=LabelEncoder().fit_transform(newdata[column])
 
numerical_feature_columns = list(newdata._get_numeric_data().columns)
numerical_feature_columns
newdata.corr() 
X=newdata.iloc[:,:-1]
Y=newdata.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.33, 
                                                    random_state=8)
model= xgb.XGBClassifier(n_estimators=400,seed=14,learning_rate=0.3)
model.fit(x_train,y_train)
 
def generate_accuracy_and_heatmap(model, x, y):
#     cm = confusion_matrix(y,model.predict(x))
#     sns.heatmap(cm,annot=True,fmt="d")
    ac = accuracy_score(y,model.predict(x))
    f_score = f1_score(y,model.predict(x))
    
    print('Accuracy is: ', ac)
    print('F1 score is: ', f_score)
    
    return 1
 
generate_accuracy_and_heatmap(model, x_test, y_test)

pickle.dump(model,open('s.pkl','wb'))
