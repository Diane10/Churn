# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 02:32:01 2020

@author:Diane Tuyizere
"""


import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,plot_confusion_matrix,plot_roc_curve,precision_score,recall_score,precision_recall_curve,roc_auc_score,auc
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pickle
 
import xgboost as xgb
 
import warnings
warnings.filterwarnings('ignore')
load_model=pickle.load(open('s.pkl','rb'))
 
 
data=pd.read_csv('https://raw.githubusercontent.com/Diane10/ML/master/Customer-Churn.csv')
 
#fetuare extracting
#q1 Using the given dataset extract the relevant features that can define a customer churn. [5]
cols=['tenure','Contract','OnlineSecurity','TechSupport','OnlineBackup','DeviceProtection','MonthlyCharges','Churn']
newdata=data[cols]
target = 'Churn'
le=LabelEncoder()
le=LabelEncoder()
 
 
newdata['Contract']=le.fit_transform(newdata['Contract'])
newdata['OnlineSecurity']=le.fit_transform(newdata['OnlineSecurity'])
newdata['TechSupport']=le.fit_transform(newdata['TechSupport'])
newdata['OnlineBackup']=le.fit_transform(newdata['OnlineBackup'])
newdata['DeviceProtection']=le.fit_transform(newdata['DeviceProtection'])
newdata['Churn']=le.fit_transform(newdata['Churn'])
 
X = newdata.loc[:, newdata.columns != target]
Y = newdata.loc[:, newdata.columns == target]
 
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.33, random_state=8)
 
from sklearn.preprocessing import StandardScaler
sl=StandardScaler()
X_trained= sl.fit_transform(X_train)
X_tested= sl.fit_transform(X_test)
 
class_name=['yes','no']
st.title("Machine Learning Assignment")
html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Custormer Churn Prediction ML App </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
 
st.markdown("""
Machine Learning models which predict potential of customer to churn
""")
st.sidebar.title('Customer churn Prediction')
 
st.sidebar.markdown("""
Machine Learning models which predict potential customer to churn
""")
 
if st.sidebar.checkbox("show raw data",False):
    st.subheader("Customer Churn for classification")
    st.write(data)
if st.sidebar.checkbox("Show Selected Feature"):
    st.write(newdata)
if st.sidebar.checkbox("Show a Statistical Analysis"):
    st.write(newdata.describe())
if st.sidebar.checkbox("Show a corration"):
    st.write(newdata.corr()) 
 
 
st.sidebar.subheader('Visualization')
if st.sidebar.checkbox("Pair plot",False):
  k = 19 #number of variables for heatmap
  cols = newdata.corr().nlargest(k, target)[target].index
  cm = newdata[cols].corr()
  fig= plt.figure(figsize=(30,16))
  st.pyplot(fig, use_container_width=True)
 
if st.sidebar.checkbox("Graph plot",False):
   fig.add_trace(go.Scatter(x=newdata.Churn, y=newdata.MonthlyCharges))
   st.pyplot(fig, use_container_width=True)
 
 
st.sidebar.subheader('Choose Classifer')
classifier_name = st.sidebar.selectbox(
    'Choose classifier',
    ('KNN', 'SVM', 'Random Forest','Logistic Regression','XGBOOST')
)
if classifier_name == 'SVM':
    st.sidebar.subheader('Model Hyperparmeter')
    c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='c')
    kernel= st.sidebar.radio("kernel",("linear","rbf"),key='kernel')
    gamma= st.sidebar.radio("gamma(kernel coefficiency",("scale","auto"),key='gamma')
 
    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
 
    if st.sidebar.button("classify",key='classify'):
        st.subheader("SVM result")
        svcclassifier= SVC(C=c,kernel=kernel,gamma=gamma)
        svcclassifier.fit(X_trained,y_train)
        y_pred= svcclassifier.predict(X_tested)
        acc= accuracy_score(y_test,y_pred)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_pred,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_pred,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('confusion matrix')
            plot_confusion_matrix(svcclassifier,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('plot_roc_curve')
            plot_roc_curve(svcclassifier,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('precision_recall_curve')
            plot_roc_curve(svcclassifier,X_tested,y_test)
            st.pyplot()
        
 
 
if classifier_name == 'Logistic Regression':
    st.sidebar.subheader('Model Hyperparmeter')
    c= st.sidebar.number_input("c(Reguralization)",0.01,10.0,step=0.01,key='Logistic')
    max_iter= st.sidebar.slider("maximum number of iteration",100,500,key='max_item')
   
 
    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
 
    if st.sidebar.button("classify",key='classify'):
        st.subheader("Logistic Regression result")
        Regression= LogisticRegression(C=c,max_iter=max_iter)
        Regression.fit(X_trained,y_train)
        y_prediction= Regression.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('confusion matrix')
            plot_confusion_matrix(Regression,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('plot_roc_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('precision_recall_curve')
            plot_roc_curve(Regression,X_tested,y_test)
            st.pyplot()
        
            
 
if classifier_name == 'Random Forest':
    st.sidebar.subheader('Model Hyperparmeter')
    n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='estimators')
    max_depth= st.sidebar.number_input("maximum depth of tree",1,20,step=1,key='max_depth')
    bootstrap= st.sidebar.radio("Boostrap sample when building trees",("True","False"),key='boostrap')
 
 
    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
 
    if st.sidebar.button("classify",key='classify'):
        st.subheader("Random Forest result")
        model= RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap)
        model.fit(X_trained,y_train)
        y_prediction= model.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('confusion matrix')
            plot_confusion_matrix(model,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('plot_roc_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('precision_recall_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot() 
 
 
if classifier_name == 'KNN':
    st.sidebar.subheader('Model Hyperparmeter')
    n_neighbors= st.sidebar.number_input("Number of n_neighbors",5,30,step=1,key='neighbors')
    leaf_size= st.sidebar.slider("leaf size",30,200,key='leaf')
    weights= st.sidebar.radio("weight function used in prediction",("uniform","distance"),key='weight')
 
 
    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
 
    if st.sidebar.button("classify",key='classify'):
        st.subheader("KNN result")
        model= KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size,weights=weights)
        model.fit(X_trained,y_train)
        y_prediction= model.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        if 'confusion matrix' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('confusion matrix')
            plot_confusion_matrix(model,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('plot_roc_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('precision_recall_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot() 
 
 
if classifier_name == 'XGBOOST':
    st.sidebar.subheader('Model Hyperparmeter')
    n_estimators= st.sidebar.number_input("Number of trees in the forest",100,5000,step=10,key='XGBestimators')
    seed= st.sidebar.number_input("number of the seed",1,150,step=1,key='seed')
    
    
    
 
 
    metrics= st.sidebar.multiselect("What is the metrics to plot?",('confusion matrix','roc_curve','precision_recall_curve'))
 
    if st.sidebar.button("classify",key='classify'):
        st.subheader("XGBOOST result")
        model= xgb.XGBClassifier(n_estimators=n_estimators,seed=seed)
        model.fit(X_trained,y_train)
        y_prediction= model.predict(X_tested)
        acc= accuracy_score(y_test,y_prediction)
        st.write("Accuracy:",acc.round(2))
        st.write("precision_score:",precision_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("recall_score:",recall_score(y_test,y_prediction,labels=class_name).round(2))
        st.write("ROC_AUC_score:",roc_auc_score(y_test,y_prediction).round(2))
 
       
 
        if 'confusion matrix' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('confusion matrix')
            plot_confusion_matrix(model,X_tested,y_test,display_labels=class_name)
            st.pyplot()
        if 'roc_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('plot_roc_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot()
        if 'precision_recall_curve' in metrics:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader('precision_recall_curve')
            plot_roc_curve(model,X_tested,y_test)
            st.pyplot() 
 
if st.sidebar.checkbox("Do you want to prediction?",key='prediction'):
    list=[]
    st.subheader('Please fill out this form')
    cols=['tenure','Contract','OnlineSecurity','TechSupport','OnlineBackup','DeviceProtection','MonthlyCharges','Churn']
    
    tenure = st.slider("what is your tenure values",1,200,key='tenure')
    list.append(tenure)
 
    Contract = st.selectbox(
    'what is your Contract?',
    ('Month-to-month', 'One year','Two year'))
    if Contract == 'Month-to-month':
      list.append(0)
    elif Contract == 'One year':
      list.append(1)  
    else:
      list.append(2)
 
    security = st.radio("Do you have online security?",('No', 'Yes', 'No internet service'))
    if security == 'Yes':
      #st.success("You are Active")
      list.append(2)
    elif security == 'No':
      #st.success("You are Active")
      list.append(0)  
    else:
      list.append(1)
      #st.warning("Inactive, Activate") 
 
    techsupport = st.radio("Do you have Tech support?",('No', 'Yes', 'No internet service'))
    if techsupport == 'Yes':
      #st.success("You are Active")
      list.append(2)
    elif techsupport == 'No':
      #st.success("You are Active")
      list.append(0)  
    else:
      list.append(1)  
 
    OnlineBackup = st.selectbox("Do you have online backup",
    ('Yes', 'No', 'No internet service'))
    if OnlineBackup == 'Yes':
      #st.success("You are Active")
      list.append(2)
    elif OnlineBackup == 'No':
      #st.success("You are Active")
      list.append(0)  
    else:
      list.append(1)
      #st.warning("Inactive, Activate")
 
    DeviceProtection = st.radio("Do you have Device Protection",("Yes","No","No internet service"))
    if DeviceProtection == 'Yes':
      #st.success("You are Active")
      list.append(2)
    elif DeviceProtection == 'No':
      #st.success("You are Active")
      list.append(0)  
    else:
      list.append(1)
      #st.warning("Inactive, Activate")  
 
    
    monthlycharge = st.text_input("what is your monthly charge","Type Here")
    list.append(monthlycharge)
 
 
    if st.button("Prediction",key='predict'):
         my_array= np.array([list])
         #model= xgb.XGBClassifier(n_estimators=400,seed=14,learning_rate=0.3)
         #model.fit(X_trained,y_train)
         y_user_prediction= load_model.predict(my_array)
         if y_user_prediction==0:
          st.subheader("a new customer can not result in a churn")
          st.balloons()
          st.write(y_user_prediction)
         elif y_user_prediction==1:
          st.subheader("a new customer can  result in a churn")
          st.balloons()