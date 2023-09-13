import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import warnings


# Suppress Python warnings to make the output clean
warnings.filterwarnings("ignore")

# Set the web page title and icon
favicon = Image.open('favicon.png')
st.set_page_config(page_title="Cardiovascular Complication Prediction App for Diabetic Patients", page_icon=favicon)

# Main header for the application
st.header("""
CARDIOVASCULAR COMPLICATION PREDICTION APP FOR DIABETIC PATIENTS
""")
# Display an image related to the application topic
image = Image.open('CardiovascularSystem.jpeg')
st.image(image, caption='Cardiovascular System', width=650)

# Sidebar header for inputting health profile
st.sidebar.header('Your Health Profile')

# Function to gather user inputs from the sidebar
def user_input_features():
    # User input for age using a slider
    age = st.sidebar.slider('Age (years)', 18, 80, 25)
    
    # User input for height using a number input box
    height = st.sidebar.number_input('Height (cm)', min_value=120.0, max_value=240.0, value=180.0)
    
    # User input for weight using a number input box
    weight = st.sidebar.number_input('Weight (kg)', min_value=40.0, max_value=200.0, value=74.0)
    
    # User input for systolic blood pressure using a slider
    ap_hi = st.sidebar.slider('Systolic blood pressure (mmHg)', 0, 240, 140)
    
    # User input for diastolic blood pressure using a slider
    ap_lo = st.sidebar.slider('Diastolic blood pressure (mmHg)', 0, 140, 80)
    
    # User input for cholesterol using a number input box
    cholesterol = st.sidebar.number_input('Cholesterol (mg/dL)', min_value=80.0, max_value=300.0, value=120.0)
    
    # Create a dictionary to store all the user inputs
    data = {'Age': age,
            'Height': height,
            'Weight': weight,
            'Systolic blood pressure': ap_hi,
            'Diastolic blood pressure': ap_lo,
            'Cholesterol': cholesterol,
            }
    
    # Convert the dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    
    return features

# Call the function and store the results in a DataFrame
df = user_input_features()

# Display a subheader for the application description
st.subheader('Description')

# Provide a detailed description of the application using markdown
st.markdown("""
This Cardiovascular Complication Prediction App for diabetic patients is designed to predict the 
risk of cardiovascular complications in patients with a history of type-2 diabetes mellitus (T2DM). 
\nDeveloped with an artificial intelligence (AI) approach, this app can predict the risk of cardiovascular complications with an **accuracy of 71%**, **_recall_ of 69%**, 
and **precision of 70%**. The results of the research and development of this application can be viewed at the following [link](https://colab.research.google.com/drive/1zH20OSkCUqXbGUgEy7Dt792-HkoFxWtF?usp=sharing).
""")

# Display a subheader for the user's health profile
st.subheader('Your Health Profile')

# Display each component of the user's health profile
st.write('**Age:** ', df['Age'][0], ' years')
st.write('**Height:** ', round(df['Height'][0], 2), ' cm')
st.write('**Weight:** ', round(df['Weight'][0], 2), ' kg')
st.write('**Systolic blood pressure:** ', df['Systolic blood pressure'][0], ' mmHg')
st.write('**Diastolic blood pressure:** ', df['Diastolic blood pressure'][0], ' mmHg')
st.write('**Cholesterol**: ', round(df['Cholesterol'][0], 2), ' mg/dL')

# Data Preprocessing

# Create BMI (Body Mass Index) Feature
def calc_bmi(weight, height):
  """Function to calculate BMI from weight and height."""
  weight = float(weight)
  height = float(height)
  height = height / 100.0  # Convert height to meters
  return round(weight / (height**2), 2)

# Make a copy of the original DataFrame
df2 = df.copy()
# Calculate BMI for each row in the DataFrame
df2['bmi'] = df2.apply(lambda x: calc_bmi(df.Weight, df.Height), axis=1)
# Drop Height and Weight columns since BMI is derived from them
df2 = df2.drop(columns=['Height', 'Weight'], errors='ignore')

# Create a feature for Hypertension Stage
dict_htstage = {
    'normal': 1,
    'stage_1': 2,
    'stage_2': 3,
    'crisis': 4
}
# Reverse mapping for the hypertension stages
dict_htstage_swap = {v: k for k, v in dict_htstage.items()}

def get_ht_stage(systole: int, diastole: int) -> int:
  """Function to determine hypertension stage based on blood pressure readings."""
  if systole < 130 and diastole < 80:
    return dict_htstage['normal']
  elif (systole >= 130 and systole < 140) or (diastole >= 80 and diastole < 90):
    return dict_htstage['stage_1']
  elif (systole >= 140 and systole < 180) or (diastole >= 90 and diastole < 120):
    return dict_htstage['stage_2']
  else:
    return dict_htstage['crisis']

# Assign hypertension stage for each row in the DataFrame
df2['ht_stage'] = df2.apply(lambda x: get_ht_stage(x['Systolic blood pressure'], x['Diastolic blood pressure']), axis=1)

# Encode cholesterol levels into categories
dict_cholesterol = {"normal": 1, "above normal": 2, "well above normal": 3}
def get_cholasterol_level(chol: int) -> int:
  """Function to categorize cholesterol levels."""
  if chol < 200:
    return dict_cholesterol['normal']
  elif (chol >= 200 and chol < 240):
    return dict_cholesterol['above normal']
  elif chol >= 240:
    return dict_cholesterol['well above normal']

# Assign cholesterol category for each row in the DataFrame
df2['Cholesterol'] = df2.apply(lambda x: get_cholasterol_level(x['Cholesterol']), axis=1)

# Rename columns for consistency
df2 = df2.rename(columns={"Systolic blood pressure": "ap_hi", "Diastolic blood pressure": "ap_lo", "Age": "age", 
"Cholesterol": "cholesterol",})

# Reorder the DataFrame columns
columns = ["age", "bmi", "ap_hi", "ap_lo", "ht_stage", "cholesterol",]
df2 = df2.reindex(columns=columns)

# Feature selection based on prior experiments (as noted)
df2 = df2[['age', 'bmi', 'ap_hi', 'ht_stage', 'cholesterol',]]


# Load the Trained Model

# Specify the path to the saved model file
pkl_filename = "cardio_clf.pkl"
# Open and load the model from the file
with open(pkl_filename, 'rb') as file:
    classifier = pickle.load(file)

# Make Predictions using the Model

# Predict the class label (0 or 1) for the input data
prediction = classifier.predict(df2)
# Predict the probabilities for each class
prediction_proba = classifier.predict_proba(df2)

# Map the predicted class label to its corresponding string label
if prediction == 1:
  prediction = 'cardio'
else:
  prediction = 'non-cardio'


# Predict and Display Results

# Define the success and danger messages to display based on prediction results
success_message = """<p style="font-family:sans-serif; color:Green;">Based on the analysis of your current health profile,
you have a lower risk of experiencing cardiovascular complications in the future. However, you still need to maintain your health
to be free from various disease complication risks. Here are some healthy living tips you can follow.</p>"""
danger_message = """<p style="font-family:sans-serif; color:Red;">Based on the analysis of your current health profile,
your risk of experiencing cardiovascular complications in the future is higher.
Consult your doctor immediately for further information. Here are some healthy living tips you can follow.</p>
"""

# Display the prediction header
st.subheader('Prediction Results')

# Display the high and low risk percentages and the corresponding messages
if round(prediction_proba[0,1], 5) > round(prediction_proba[0,0], 5):
  st.write("**High risk of cardiovascular complications:** ", round(prediction_proba[0,1]*100, 2), "%")
  st.write("Low risk of cardiovascular complications: ",  round(prediction_proba[0,0]*100, 2), "%")
  st.markdown(danger_message, unsafe_allow_html=True)
  st.markdown(
  """
  - Healthy diet
    - Choose high fiber foods (fruits / vegetables)
    - Reduce sugar and simple carbohydrate consumption
    - Limit salt intake
    - Reduce consumption of fatty foods
    - Do not consume alcohol
  - Exercise and always be active
  - Manage stress
  - Check blood sugar regularly
  - Take medications regularly as prescribed by the doctor

  """
  )
  st.markdown("""Remember! Your healthy life is a healthy life for yourself and your family. 
  Visit your doctor immediately for further health checks.
  \n#StayHealthy #HealthIsBeautiful #LiveHealthyWithDiabetes""")

elif round(prediction_proba[0,1], 5)*100 < round(prediction_proba[0,0], 5)*100:
  st.write("**Low risk of cardiovascular complications:** ",  round(prediction_proba[0,0]*100, 2), "%")
  st.write("High risk of cardiovascular complications: ", round(prediction_proba[0,1]*100, 2), "%")
  st.markdown(success_message, unsafe_allow_html=True)
  st.markdown(
  """
  - Healthy diet
    - Choose high fiber foods (fruits / vegetables)
    - Reduce sugar and simple carbohydrate consumption
    - Limit salt intake
    - Reduce consumption of fatty foods
    - Do not consume alcohol
  - Exercise and always be active
  - Manage stress
  - Check blood sugar regularly
  - Take medications regularly as prescribed by the doctor

  """
  )

# Style to hide the default Streamlit menu and display a custom footer
hide_menu = """
  <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: visible;}
    footer:after{
      content:'Copyright © 2022 Fahmi Sajid (23522028) and Arief Purnama Muharram (23521013), STEI ITB';
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

# Apply the custom style to the Streamlit app
st.markdown(hide_menu, unsafe_allow_html=True)

