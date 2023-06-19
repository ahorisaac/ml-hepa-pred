# -- import the dependencies 
import streamlit as st
import time as t

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score


# -- loading the data from csv to pandas DataFrame 
hepa = pd.read_csv("./datasets/HepatitisCdata.csv")

# -- clean SEX column
hepa["Sex"].replace({'m':0, 'f':1}, inplace=True)

# -- data cleaning 
to_be_cleaned = ["PROT", "ALB", "ALT", "ALP", "CHOL"]
for i in to_be_cleaned:
    hepa[i].fillna(round(hepa[i].mean(), 1), inplace=True)
    
hepa = hepa.drop('Unnamed: 0', axis=1)

# -- clean Category column
hepa["Category"].replace({'0=Blood Donor':0, '0s=suspect Blood Donor':0, '1=Hepatitis':1, '2=Fibrosis':1, '3=Cirrhosis':1}, inplace=True)

# -- separating the features and target

X = hepa.drop('Category', axis=1)
y = hepa['Category']

# -- features and target 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create AdaBoost classifier and fit the model on training dataset
adb_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
adb_clf.fit(X_train, y_train)

# make predictions on testing dataset
y_pred = adb_clf.predict(X_test)

# evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("AdaBoost Prediction -> Accuracy : {}".format(accuracy))

# -- making a predictive system 
def pred_hepa(age, gender, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot):
    input_data = (age, gender, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot,)

    # -- change the input data to numpy array 
    input_data_as_nparr = np.asarray(input_data)

    # -- reshape the array as we are predicting for one instance 
    input_data_reshaped = input_data_as_nparr.reshape(1, -1)

    prediction = adb_clf.predict(input_data_reshaped)

    st.write("DIAGNOSIS COMPLETE :100:")
    if (prediction[0] == 0):
        st.success("The patient has no HEPATITIS :white_check_mark:")
        return True
    else:
        st.error("The patient has HEPATITIS :heavy_exclamation_mark:")
        return False

# -- web application (code)

def main():
    # -- hepa features input
    with st.columns(3)[1]:
        st.image("./images/icon-1.png")

    st.title("Hepatitis Prediction Application")

    st.subheader("Patient Medical Form")

    with st.form("liver_diag_form", clear_on_submit=False):
        age = st.text_input("Age")

        gender = st.radio("Gender", options=("Male", "Female"))
        
        alb = st.number_input("Albumin")
        
        alp = st.number_input("Alkaline Phosphotase")
        
        alt = st.number_input("Alanine Aminotransferase")
        
        ast = st.number_input("Aspartate Aminotransferase")

        bil = st.number_input("Bilirubin")
        
        che = st.number_input("Serum Cholinesterase")

        chol = st.number_input("Serum Cholesterol and Chronic")

        crea = st.number_input("Creatinine")

        ggt = st.number_input("Gamma-Glutamyl Transferase")

        prot = st.number_input("Protein")

        # -- prediction, form submit button 
        submitted = st.form_submit_button("Predict", type="primary", help="click to predict hepatitis")
        
        if submitted:
            age = int(age)
            
            if gender == "Male":
                gender = 0
            else:
                gender = 1
            
            pred_hepa(age, gender, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot)


if __name__ == "__main__":
    main()
