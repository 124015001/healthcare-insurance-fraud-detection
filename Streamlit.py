from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score,precision_score, recall_score
import pickle
import seaborn as sns

# def main():
st.title("medical aid claims")
st.sidebar.title("Choose Classifier")
# st.sidebar.markdown("Choose Classifier")
# st.sidebar.subheader("Choose classifier")

# if __name__ == '_main_':
#     main()


df=pd.read_csv('medical_aid_claims.csv')


if st.sidebar.checkbox("Display data", False):
    st.subheader("European Credit Card Fraud dataset")
    st.write(df)





@st.cache_data(persist=True)
def split(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)
    x_test = pickle.load(open('X_test.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))
    x_train = pickle.load(open('X_train.pkl', 'rb'))
    y_train = pickle.load(open('y_train.pkl', 'rb'))
    
    
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split(df)
# st.write(x_test.shape)



# st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ( "Logistic Regression","Random Forest"))

class_names = ["Fraud", "Not Fraud"]
def roc_print(ROC):
    fpr,tpr,_ =ROC
    plt.figure(figsize=(10,5))
    plt.grid()
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.plot(fpr,tpr,linestyle='-',label=classifier)
def confusionMatrix(cm):
    conf_matrix = cm
    print("Confusion Matrix for \n",conf_matrix)
    nb=sns.heatmap(conf_matrix,fmt='d',annot=True,cmap="viridis_r")
    nb.set_xlabel("Actual Label",fontsize=14, labelpad=10)
    nb.set_ylabel("Prediction Label", fontsize=14, labelpad=10)
    nb.set(title='Confusion Matrix')
def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test,y_pred)#, labels= class_names
        confusionMatrix(conf_matrix)
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        rc = roc_curve(y_test,y_pred)
        roc_print(rc)
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        precision_recall_curve(y_test,y_pred)
        st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)


if classifier == "Logistic Regression":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = pickle.load(open('lr.pkl', 'rb'))
        st.write(x_test.shape)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3))
        plot_metrics(metrics)

        
if classifier == "Random Forest":
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        # model = LogisticRegression(random_state=0)
        # model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        # model.fit(x_train, y_train)
        model = pickle.load(open('rfc.pkl', 'rb'))
        st.write(x_test.shape)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3))
        plot_metrics(metrics)
        