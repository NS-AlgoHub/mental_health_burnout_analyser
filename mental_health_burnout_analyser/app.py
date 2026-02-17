import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title='Mental Health Burnout Predictor')

with open('model/burnout_model.pkl','rb') as f:
    model, scaler = pickle.load(f)

st.title('Mental Health Burnout Risk Predictor')
st.write('Demo system to assess burnout risk in working professionals')

work_hours = st.slider('Work hours per day', 6, 14, 9)
sleep = st.slider('Sleep hours per day', 4, 9, 7)
stress = st.slider('Stress level (1-10)', 1, 10, 5)
screen = st.slider('Screen time per day (hours)', 3, 14, 6)
exercise = st.slider('Exercise days per week', 0, 6, 2)

if st.button('Check Burnout Risk'):
    data = np.array([[work_hours, sleep, stress, screen, exercise]])
    data_scaled = scaler.transform(data)
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    if pred == 1:
        st.error(f'High Burnout Risk (Probability: {prob:.2f})')
        st.info('Consider reducing work hours, improving sleep, and taking breaks')
    else:
        st.success(f'Low Burnout Risk (Probability: {prob:.2f})')
        st.info('Maintain healthy work-life balance')

st.caption('Demo prediction system – not a medical diagnosis')
