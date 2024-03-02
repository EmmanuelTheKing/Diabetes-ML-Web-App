# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np

import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/user/Documents/Model Deployement/trained_model.sav', 'rb'))

input_data=(2,197,70,45,543,30.5,0.158,53)

#changing the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instace
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
    
    
    
else:
    print("The person is diabetic")
