import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import warnings


# Suppress warnings
warnings.filterwarnings("ignore")

# Web title
favicon = Image.open('favicon.png')
st.set_page_config(page_title="Aplikasi Prediksi Risiko Komplikasi Kardiovaskuler untuk Pasien Diabetes", page_icon=favicon)

# Header
st.header("""
APLIKASI PREDIKSI RISIKO KOMPLIKASI KARDIOVASKULER UNTUK PASIEN DIABETES
""")
image = Image.open('CardiovascularSystem.jpeg')
st.image(image, caption='Cardiovascular System', width=650)
st.sidebar.header('Profil kesehatan Anda')

# User input
def user_input_features():
    age = st.sidebar.slider('Umur (tahun)', 18, 80, 25)
    height = st.sidebar.number_input('Tinggi badan (cm)', min_value=120.0, max_value=240.0, value=180.0)
    weight = st.sidebar.number_input('Berat badan (kg)', min_value=40.0, max_value=200.0, value=74.0)
    ap_hi = st.sidebar.slider('Tekanan darah sistolik (mmHg)', 0, 240, 140)
    ap_lo = st.sidebar.slider('Tekanan darah diastolik (mmHg)', 0, 140, 80)
    cholesterol = st.sidebar.number_input('Kolesterol (mg/dL)', min_value=80.0, max_value=300.0, value=120.0)
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

# Application description
st.subheader('Deskripsi')
st.markdown("""
Aplikasi prediksi risiko komplikasi kardiovaskuler untuk pasien diabetes merupakan aplikasi yang digunakan untuk memprediksi 
risiko komplikasi kardiovaskuler pada pasien dengan riwayat diabetes melitus tipe-2 (DMT2). 
\nAplikasi ini dikembangkan dengan pendekatan berbasis artificial intelligence (AI). Aplikasi ini mampu memprediksi risko komplikasi kardiovaskuler dengan **akurasi 71%**, **_recall_ 69%**, 
dan **presisi 70%**. Hasil penelitian pengembangan aplikasi tersebut dapat dilihat pada link [berikut](https://docs.google.com/document/d/1weWAHR_an6XnVmIxkZXP2MJyso4JFOfpGlTuE0rhV0A/edit?usp=sharing).""")

# User's profile
st.subheader('Profil kesehatan Anda')
st.write('**Umur:** ', df['Age'][0], ' tahun')
st.write('**Tinggi badan:** ',round(df['Height'][0],2), ' cm')
st.write('**Berat badan:** ',round(df['Weight'][0],2), ' kg')
st.write('**Tekanan darah sistolik:** ',df['Systolic blood pressure'][0], ' mmHg')
st.write('**Tekanan darah diastolik:** ',df['Diastolic blood pressure'][0], ' mmHg')
st.write('**Kolesterol**: ',round(df['Cholesterol'][0],2), ' mg/dL')

# Preprocessing
# Create BMI Feature
def calc_bmi(weight, height):
  weight = float(weight)
  height = float(height)
  height = height / 100.0
  return round(weight / (height**2), 2)
df2 = df.copy()
df2['bmi'] = df2.apply(lambda x: calc_bmi(df.Weight, df.Height), axis=1)
df2 = df2.drop(columns=['Height', 'Weight'], errors='ignore')

# Create Hypertension Stage Feature
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

# Encode cholesterol
dict_cholesterol = {"normal": 1, "above normal": 2, "well above normal":3}
def get_cholasterol_level(chol: int) -> int:
  if chol < 200:
    return dict_cholesterol['normal']
  elif (chol >= 200 and chol < 240):
    return dict_cholesterol['above normal']
  elif chol >= 240:
    return dict_cholesterol['well above normal']
df2['Cholesterol'] = df2.apply(lambda x: get_cholasterol_level(x['Cholesterol']), axis=1)
df2 = df2.rename(columns={"Systolic blood pressure": "ap_hi", "Diastolic blood pressure": "ap_lo", "Age":"age", 
"Cholesterol":"cholesterol",})

# Reorder the DataFrame
columns = ["age", "bmi", "ap_hi", "ap_lo", "ht_stage", "cholesterol",]
df2 = df2.reindex(columns=columns)

# Feature selection
# Why this? See experiment notebook for details
df2 = df2[['age', 'bmi', 'ap_hi', 'ht_stage', 'cholesterol',]]

# Load Model
pkl_filename = "cardio_clf.pkl"
with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)

# Prediction
prediction = classifier.predict(df2)
prediction_proba = classifier.predict_proba(df2)
if prediction == 1:
  prediction = 'cardio'
else: prediction = 'bukan cardio'

#Predict Result
success_message = """<p style="font-family:sans-serif; color:Green;">Berdasarkan hasil analisis terhadap profil kesehatan Anda saat ini, 
Anda memiliki risiko yang lebih rendah untuk mengalami risiko komplikasi kardiovaskuler di kemudian hari. Walaupun demikian, Anda tetap perlu menjaga kondisi kesehatan 
Anda agar Anda terbebas dari berbagai risiko komplikasi penyakit. Berikut merupakan beberapa tips hidup sehat yang dapat Anda lakukan.</p>"""
danger_message = """<p style="font-family:sans-serif; color:Red;">Berdasarkan hasil analisis terhadap profil kesehatan Anda saat ini, 
risiko Anda untuk mengalami komplikasi kardiovaskuler di kemudian hari lebih tinggi. 
Segera konsultasikan dengan dokter Anda untuk mendapatkan informasi lebih lanjut. Berikut merupakan beberapa tips hidup sehat yang dapat Anda lakukan.</p>
"""
st.subheader('Hasil Prediksi')
if round(prediction_proba[0,1], 5)> round(prediction_proba[0,0], 5):
  st.write("**Risiko tinggi mengalami komplikasi kardiovaskuler:** ", round(prediction_proba[0,1]*100, 2), "%")
  st.write("Risiko rendah mengalami komplikasi kardiovaskuler: ",  round(prediction_proba[0,0]*100, 2), "%")
  st.markdown(danger_message, unsafe_allow_html=True)
  st.markdown(
  """
  - Diet sehat
    - Pilih makanan berserat tinggi (buah-buahan / sayur-sayuran)
    - Kurang konsumsi gula dan karbohidrat sederhana
    - Batasi konsumsi garam-garaman
    - Kurangi konsumsi makanan berlemak
    - Tidak mengonsumsi alkohol
  - Olahraga dan selalu aktif
  - Kelola stress
  - Perika gula darah secara rutin
  - Konsumsi obat-obatan secara rutin sesuai dengan anjuran dokter

  """
  )
  st.markdown("""Ingat! Hidup sehat Anda adalah hidup sehat diri sendiri dan keluarga. 
  Segera kunjungi dokter Anda untuk mendapatkan pemeriksaan kesehatan lebih lanjut.
  \n#SalamSehat #SehatItuIndah #HidupSehatDenganDiabetes""")
elif round(prediction_proba[0,1], 5)*100 < round(prediction_proba[0,0], 5)*100:
  st.write("**Risiko rendah mengalami komplikasi kardiovaskuler:** ",  round(prediction_proba[0,0]*100, 2), "%")
  st.write("Risiko tinggi mengalami komplikasi kardiovaskuler: ", round(prediction_proba[0,1]*100, 2), "%")
  st.markdown(success_message, unsafe_allow_html=True)
  st.markdown(
  """
  - Diet sehat
    - Pilih makanan berserat tinggi (buah-buahan / sayur-sayuran)
    - Kurang konsumsi gula dan karbohidrat sederhana
    - Batasi konsumsi garam-garaman
    - Kurangi konsumsi makanan berlemak
    - Tidak mengonsumsi alkohol
  - Olahraga dan selalu aktif
  - Kelola stress
  - Perika gula darah secara rutin
  - Konsumsi obat-obatan secara rutin sesuai dengan anjuran dokter

  """
  )
  
hide_menu = """
  <style>
    #MainMenu {visibility: visible;}
    footer {visibility: visible;}
    footer:after{
      content:'Copyright © 2022 Fahmi Sajid (23522028) dan Arief Purnama Muharram (23521013), STEI ITB';
      display:block;
      position:relative;
      padding-top:5px;
      padding-bottom:5px;
      padding-left:0px;
      padding-right:0px;
      top:3px;
    }
  </style>
"""

st.markdown(hide_menu, unsafe_allow_html=True)
