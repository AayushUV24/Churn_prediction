import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

# Load the trained model,Scaler pickel,onehot
model = tf.keras.models.load_model('model.h5')
# load the encoder and scaler
with open('label_encoder_gender.pkl','rb')as file:
    label_encoder_gender = pickle.load(file)
with open('Ohe.pkl','rb')as file:
    Ohe = pickle.load(file)
with open('scaler.pkl','rb')as file:
    scaler = pickle.load(file)  

## Streamlit app
st.title("Customer Churn Prediction") 

#User Input
geography = st.selectbox('Geography',Ohe.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',15,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimeted_scalary = st.number_input('Estimeted Scalary')
tenure = st.slider('Tenure',0,10)
num_of_prodeucts = st.slider('Numder of products',1,4)
has_Cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member  = st.selectbox('Is Active Member',[0,1])


input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],     
    'Age': [age], 
    'Tenure':[tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_prodeucts],
    'HasCrCard': [has_Cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimeted_scalary]
})

geo_encoded = Ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=Ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn probability:{prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')