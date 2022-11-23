import streamlit as st
import pandas as pd
from PIL import Image
import pickle

image = Image.open('CardiovascularSystem.jpeg')

st.header("""
PREDIKSI RISIKO KOMPLIKASI KARDIOVASKULER PADA PASIEN DIABETES
""")
st.image(image, caption='Cardiovascular System', width=650)
st.sidebar.header('User Input Parameter')

def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 25)
    height = st.sidebar.slider('Height', 80, 200, 160)
    weight = st.sidebar.slider('Weight', 10, 200, 80)
    ap_hi = st.sidebar.slider('Systolic blood pressure', 10, 140, 114)
    ap_lo = st.sidebar.slider('Diastolic blood pressure', 0, 90, 70)
    cholesterol = st.sidebar.selectbox('Cholesterol',('normal','above normal','well above normal'))
    data = {'Age': age,
            'Height': height,
            'Weight': weight,
            'Systolic blood pressure':ap_hi,
            'Diastolic blood pressure':ap_lo,
            'Cholesterol': cholesterol,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#Preprocess
#Create BMI Feature
def calc_bmi(weight, height):
  weight = float(weight)
  height = float(height)
  height = height / 100.0
  return round(weight / (height**2), 2)

df2 = df.copy()
df2['bmi'] = df2.apply(lambda x: calc_bmi(df.Weight, df.Height), axis=1)

df2 = df2.drop(columns=['Height', 'Weight'], errors='ignore')

#Create Hypertension Stage Feature
dict_htstage = {
    'normal': 1,
    'stage_1': 2,
    'stage_2': 3,
    'crisis': 4
}

dict_htstage_swap = {v: k for k, v in dict_htstage.items()}

def get_ht_stage(systole: int, diastole: int) -> int:
  if systole < 130 and diastole < 80:
    return dict_htstage['normal']
  elif (systole >= 130 and systole < 140) or (diastole >= 80 and diastole < 90):
    return dict_htstage['stage_1']
  elif (systole >= 140 and systole < 180) or (diastole >= 90 and diastole < 120):
    return dict_htstage['stage_2']
  else:
    return dict_htstage['crisis']

df2['ht_stage'] = df2.apply(lambda x: get_ht_stage(x['Systolic blood pressure'], x['Diastolic blood pressure']), axis=1)

#Encode
dict_cholesterol = {"normal": 1, "above normal": 2, "well above normal":3}
df2['Cholesterol'] = df2['Cholesterol'].apply(lambda x: dict_cholesterol[x])

#rename
df2 = df2.rename(columns={"Systolic blood pressure": "ap_hi", "Diastolic blood pressure": "ap_lo", "Age":"age", 
"Cholesterol":"cholesterol",})

#reorder
columns = ["age", "bmi", "ap_hi", "ap_lo", "ht_stage", "cholesterol",]
df2 = df2.reindex(columns=columns)

#feature selection
df2 = df2[['age', 'bmi', 'ap_hi', 'ht_stage', 'cholesterol',]]

#Load Model
pkl_filename = "cardio_clf.pkl"

with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)


prediction = classifier.predict(df2)
prediction_proba = classifier.predict_proba(df2)

if prediction == 1:
  prediction = 'cardio'
else: prediction = 'bukan cardio'

#Predict Result
st.subheader('Prediction')
st.text("Emosi:")
if round(prediction_proba[0,1], 5)> round(prediction_proba[0,0], 5):
  st.write("**Probability terkena penyakit Cardiovascular:** ", round(prediction_proba[0,1]*100, 5), "%")
  st.write("Probability Sehat: ",  round(prediction_proba[0,0]*100, 3), "%")
elif round(prediction_proba[0,1], 5)*100 < round(prediction_proba[0,0], 5)*100:
  st.write("**Probability Sehat:** ",  round(prediction_proba[0,0]*100, 5), "%")
  st.write("Probability terkena penyakit Cardiovascular: ", round(prediction_proba[0,1]*100, 5), "%")
  