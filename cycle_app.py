import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Cycle Flow

This app predicts the **Estimated Day of ** Ovulation!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](sample.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        
        
        LengthofCycle = st.sidebar.slider('Cycle length', 10,50,23)
        LengthofLutealPhase = st.sidebar.slider('luetal phase', 0,30,7)
        LengthofMenses = st.sidebar.slider('Menses depth', 0,20,10)
        
        
        data = {
                'LengthofCycle': LengthofCycle,
                'LengthofLutealPhase': LengthofLutealPhase,
                'LengthofMenses': LengthofMenses
                
                }

        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire  dataset
# This will be useful for the encoding phase
cycle_raw = pd.read_csv("clean_dataset.csv")
cycle = cycle_raw.drop(columns=['EstimatedDayofOvulation'], axis=1)
df = pd.concat([input_df,cycle],axis=0)

# Encoding of ordinal features  species

encode = ['LengthofCycle','LengthofLutealPhase']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded.')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('cycle_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
EstimatedDayofOvulation = np.array([17, 15, 16, 14, 18, 12, 19, 11, 13, 27, 22,  8, 20, 21, 23, 10, 26,24, 29,  9, 25, 28,  6])
st.write(EstimatedDayofOvulation[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
