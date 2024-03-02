# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:24:57 2024

@author: user
"""

import numpy as np
import streamlit as st
import pickle
import sklearn

# loading the saved model
loaded_model = pickle.load(open('C:/Users/user/Documents/Model Deployement/trained_model.sav', 'rb'))

#Creating a function for prediction

def diabetes_prediction(input_data):
    input_data=(2,197,70,45,543,30.5,0.158,53)

    #changing the input_data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the array as we are predicting for one instace
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

def main():
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    
    #  giving the input data from the user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis =''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Results'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagnosis)
    
    
    
    
if __name__=='__main__':
    main()
    