import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
image = Image.open('CardiovascularSystem.jpeg')

st.header("""
PREDIKSI RISIKO KOMPLIKASI KARDIOVASKULER PADA PASIEN DIABETES
""")
st.image(image, caption='Cardiovascular System', width=650)
st.sidebar.header('User Input Parameter')

def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 50)
    gender = st.sidebar.selectbox('Gender',('Male','Female'))
    ap_hi = st.sidebar.slider('Systolic blood pressure', 10, 140, 70)
    ap_lo = st.sidebar.slider('Diastolic blood pressure', 0, 90, 45)
    cholesterol = st.sidebar.selectbox('Cholesterol',('normal','above normal','well above normal'))
    smoke = st.sidebar.selectbox('Smoking',('yes','no'))
    alco = st.sidebar.selectbox('Alcohol intake',('yes','no'))
    active = st.sidebar.selectbox('Physical activity',('yes','no'))
    data = {'Age': age,
            'Gender': gender,
            'Systolic blood pressure':ap_hi,
            'Diastolic blood pressure':ap_lo,
            'Cholesterol': cholesterol,
            'Smoking': smoke,
            'Alcohol intake':alco,
            'Physical activity':active,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

st.subheader('Prediction')
st.write('cardio: 90%')
st.write('not_cardio: 10%')