import streamlit as st
import pickle as pkl
import pandas as pd

model = pkl.load(open('model.pkl', 'rb'))

encoder = pkl.load(open('encoder.pkl', 'rb'))
featureEncoder = pkl.load(open('feature11.pkl', 'rb'))

st.title('Weather Prediction Model')

tab1, tab2 = st.tabs(['Weather Prediction Model', 'Weather Prediction via Upload'])

with tab1:
    precipitation = st.number_input('Precipitation')
    tempMax = st.number_input('Temp Max')
    tempMin = st.number_input('Temp Min')
    windSpeed = st.number_input('Wind Speed')
    
    year = st.selectbox("Year", list(range(2012, 2017)))
    month = st.selectbox("Month", ["Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"])
    day = st.selectbox("Day", list(range(1, 32)))
    
    button1 = st.button('Predict')

    if button1:
        if precipitation < 0:
            st.error('Precipitation cannot be negative')
        elif tempMax < 0:
            st.error('Temp Max cannot be less than 0')
        elif tempMin < -0.5:
            st.error('Temp Min cannot be less than -0.5')
        elif windSpeed >= 10:
            st.error('Wind Speed cannot be greater than 10')
        else:
            dataWeather = pd.DataFrame({
                "precipitation": [precipitation],
                "temp_max": [tempMax],
                "temp_min": [tempMin],
                "wind": [windSpeed],
                "Year": [year],
                "Month": [month],
                "Day": [day],
            })
            encodeFeatures = featureEncoder.transform(dataWeather)
            predict = model.predict(encodeFeatures)
            target=encoder.inverse_transform(predict)
            dataWeather['Predicted Weather'] = target
            st.write(dataWeather)
with tab2:
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        dataWeather = pd.read_csv(uploaded_file)
        encodeFeatures = featureEncoder.transform(dataWeather)
        predict = model.predict(encodeFeatures)
        target=encoder.inverse_transform(predict)
        dataWeather['Predicted Weather'] = target
        st.write(dataWeather)