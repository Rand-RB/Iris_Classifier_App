#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 17:27:54 2022

@author: ranood
"""
import streamlit as st
import pickle
from PIL import Image
import pandas as pd

#load the model
km = pickle.load(open('Kmeans_Classifier.pkl', 'rb'))

#load the imges
setosa= Image.open('setosa.jpg')
versicolor= Image.open('versicolor.jpg')
virginica = Image.open('virginica.jpg')


st.write("""
         # IRIS Classificatin App
         
### Let's Predict the Flowers Type Based on Some Features by Using Machine Learning
         
         """)
         
#read the data from the user
st.sidebar.header("User Input Features")
def user_input():
    sep_l = st.sidebar.slider("sepal_length", 0.0, 1.0)
    sep_w = st.sidebar.slider("sepal_width", 0.0, 1.0)
    pet_l = st.sidebar.slider("petal_length", 0.0, 1.0)
    pet_w = st.sidebar.slider("petal_width", 0.0, 1.0)
    
    features = {"sepal_length": sep_l, "sepal_width":sep_w,
                "petal_length":pet_l, "petal_width":pet_w}
    data = pd.DataFrame(features, index=[0])
    return data

user_input = user_input() 

if st.button("Click Here to Predict"):
         pred = km.predict(user_input)
         if pred == 0:
                  st.image(setosa)
         elif(pred == 1):
                  st.image(versicolor) 
         else:
                  st.image(virginica)


#print the user input
st.subheader("User Input")
st.write(user_input)
