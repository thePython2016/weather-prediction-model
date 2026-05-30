import streamlit as st
import pickle as pkl
import pandas as pd

model          = pkl.load(open('model.pkl',       'rb'))
encoder        = pkl.load(open('encoder.pkl',     'rb'))
featureEncoder = pkl.load(open('feature11.pkl',   'rb'))
numCols        = pkl.load(open('numCols.pkl',     'rb'))
bounds         = pkl.load(open('bounds.pkl',      'rb'))

st.set_page_config(page_title="Weather Prediction", layout="wide")

st.markdown("""
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }

header[data-testid="stHeader"]         { display: none !important; }
footer                                  { display: none !important; }
[data-testid="stToolbar"]              { display: none !important; }
[data-testid="stDecoration"]           { display: none !important; }
[data-testid="stStatusWidget"]         { display: none !important; }

html, body { margin: 0 !important; padding: 0 !important; }

.stApp                                          { margin: 0 !important; padding: 0 !important; }
[data-testid="stAppViewContainer"]              { margin: 0 !important; padding: 0 !important; }
[data-testid="stAppViewBlockContainer"]         { margin: 0 !important; padding: 0 !important; }
[data-testid="stMain"]                          { margin: 0 !important; padding: 0 !important; }
[data-testid="stMainBlockContainer"]            { margin: 0 !important; padding: 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"]    { margin: 0 !important; padding: 0 !important; }
[data-testid="stVerticalBlock"]                 { margin: 0 !important; padding: 0 !important; gap: 0 !important; }

.block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    width: 100% !important;
}

[data-testid="stHorizontalBlock"] {
    align-items: flex-start !important;
    gap: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}

[data-testid="stHorizontalBlock"] > div:first-child {
    background-color: #f0f2f6 !important;
    padding: 2rem 1.5rem !important;
    margin: 0 !important;
    min-height: 100vh !important;
    border-right: 1px solid #dde0e6;
}

[data-testid="stHorizontalBlock"] > div:last-child {
    padding: 4rem 2rem 2rem 2rem !important;
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Month label → int mapping (string shown to user, int sent to model)
MONTH_MAP = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4, "May":5, "Jun":6,
    "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12
}

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("<h1 style='text-align:center; margin-bottom:0.5rem;'>Weather Prediction Model</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Manual Input")

    precipitation = st.slider("Precipitation", min_value=0.0, max_value=50.0, step=0.1)
    tempMax       = st.slider("Temp Max (°C)",  min_value=0.0, max_value=50.0, step=0.1)
    tempMin       = st.slider("Temp Min (°C)",  min_value=-0.5, max_value=30.0, step=0.1)
    windSpeed     = st.slider("Wind Speed",     min_value=0.0, max_value=9.9,  step=0.1)

    year  = st.select_slider("Year",  options=list(range(2012, 2017)))
    month = st.select_slider("Month", options=list(MONTH_MAP.keys()))  # shows "Jan","Feb"...
    day   = st.select_slider("Day",   options=list(range(1, 32)))

    if st.button("Predict", use_container_width=True):
        dataWeather = pd.DataFrame({
            "precipitation": [precipitation],
            "temp_max":      [tempMax],
            "temp_min":      [tempMin],
            "wind":          [windSpeed],
            "Year":          [year],
            "Month":         [MONTH_MAP[month]],  # "Jan" → 1, "Feb" → 2 etc.
            "Day":           [day],
        })
        for col in numCols:
            dataWeather[col] = dataWeather[col].clip(lower=bounds[col]['lower'], upper=bounds[col]['upper'])
        predict      = model.predict(dataWeather)
        target       = encoder.inverse_transform(predict)
        dataWeather["Predicted Weather"] = target
        dataWeather["Month"] = month   # show string label back in results
        st.success("Prediction complete!")
        st.dataframe(dataWeather, use_container_width=True)

with col_right:
    st.subheader("Batch Prediction via CSV Upload")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if st.button("Predict from File", use_container_width=True):
        if uploaded_file is not None:
            dataWeather = pd.read_csv(uploaded_file)
            # If CSV has string months, convert to int
            if dataWeather["Month"].dtype == object:
                dataWeather["Month"] = dataWeather["Month"].map(MONTH_MAP)
            for col in numCols:
                dataWeather[col] = dataWeather[col].clip(lower=bounds[col]['lower'], upper=bounds[col]['upper'])
            predict      = model.predict(dataWeather)
            target       = encoder.inverse_transform(predict)
            dataWeather["Predicted Weather"] = target
            st.dataframe(dataWeather, use_container_width=True)
        else:
            st.warning("Please upload a CSV file before predicting.")