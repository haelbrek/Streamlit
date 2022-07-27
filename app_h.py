import streamlit as st
import pandas as pd
import numpy as np
import pickle
  
st.title('Welcome to Diabetes Prediction Application using Machine Learning Algorithms') 

file1 = open('diabetes_prediction.pkl', 'rb')
rf = pickle.load(file1)
file1.close()


data = pd.read_csv("diabete_population.csv")

print(data)
  
age = st.number_input("Enter your age") 

grossesses = st.number_input("Enter your grossesses") 

insuline= st.number_input("Enter your insuline") 
if(st.button('Predict Diabete')): 
    query = np.array([grossesses, age, insuline])

    query = query.reshape(1, 3)
    print(query)
    prediction = rf.predict(query)[0]
    st.title("Predicted value " +
             str(prediction)) 