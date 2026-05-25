import streamlit as st
import pickle as pkl
import pandas as pd

model = pkl.load(open('model.pkl', 'rb'))
encoder = pkl.load(open('encoder.pkl', 'rb'))
featureEncoder = pkl.load(open('feature11.pkl', 'rb'))

st.set_page_config(page_title="Weather Prediction", layout="wide")

st.markdown("""
<style>

.block-container {
    padding-top: 1rem !important;
}

/* Align columns to top */
[data-testid="stHorizontalBlock"] {
    align-items: flex-start;
}

/* Grey background on left column only */
[data-testid="stHorizontalBlock"] > div:first-child {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1.5rem 1.5rem 2rem 1.5rem;
    min-height: 85vh;
}

/* Push right column content down */
[data-testid="stHorizontalBlock"] > div:last-child {
    padding-top: 4rem;
    padding-left: 2rem;
}
</style>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

# ── LEFT: Title + Sliders ─────────────────────────────────────────────────────
with col_left:
    
    st.markdown("---")
    st.subheader("Manual Input")

    precipitation = st.slider("Precipitation", min_value=0.0, max_value=50.0, step=0.1)
    tempMax       = st.slider("Temp Max (°C)",  min_value=0.0, max_value=50.0, step=0.1)
    tempMin       = st.slider("Temp Min (°C)",  min_value=-0.5, max_value=30.0, step=0.1)
    windSpeed     = st.slider("Wind Speed",     min_value=0.0, max_value=9.9,  step=0.1)

    year  = st.select_slider("Year",  options=list(range(2012, 2017)))
    month = st.select_slider("Month", options=["Jan","Feb","Mar","Apr","May","Jun",
                                                "Jul","Aug","Sep","Oct","Nov","Dec"])
    day   = st.select_slider("Day",   options=list(range(1, 32)))

    if st.button("Predict", use_container_width=True):
        dataWeather = pd.DataFrame({
            "precipitation": [precipitation],
            "temp_max":      [tempMax],
            "temp_min":      [tempMin],
            "wind":          [windSpeed],
            "Year":          [year],
            "Month":         [month],
            "Day":           [day],
        })
        encodeFeatures = featureEncoder.transform(dataWeather)
        predict        = model.predict(encodeFeatures)
        target         = encoder.inverse_transform(predict)
        dataWeather["Predicted Weather"] = target
        st.success("Prediction complete!")
        st.dataframe(dataWeather, use_container_width=True)

# ── RIGHT: CSV Upload ─────────────────────────────────────────────────────────
with col_right:
    st.subheader("Batch Prediction via CSV Upload")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if st.button("Predict from File", use_container_width=True):
        if uploaded_file is not None:
            dataWeather    = pd.read_csv(uploaded_file)
            encodeFeatures = featureEncoder.transform(dataWeather)
            predict        = model.predict(encodeFeatures)
            target         = encoder.inverse_transform(predict)
            dataWeather["Predicted Weather"] = target
            st.dataframe(dataWeather, use_container_width=True)
        else:
            st.warning("Please upload a CSV file before predicting.")