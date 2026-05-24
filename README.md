#  Weather Prediction Model

A machine learning web application that predicts weather conditions from uploaded CSV data, built with Streamlit and XGBoost Classifier.

The primary focus of the model is predicting whether the weather condition will be Rain or Sun based on meteorological features.

🔗 Live Demo: https://myweatherpredictionmodel.streamlit.app/

---

# Model Performance

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 0.8018 |
| Weighted Avg F1 | 0.7715 |
| Rain F1-Score   | 0.9287 |
| Sun F1-Score    | 0.8266 |

> The model achieves strong performance in predicting rain and sunny weather conditions using meteorological data such as precipitation, temperature, and wind speed.



## Model Details

* Algorithm: XGBoost Classifier (`XGBClassifier`)

* Preprocessing: Feature transformation pipeline

* Input Features:

  * Precipitation
  * Maximum Temperature
  * Minimum Temperature
  * Wind Speed

* Output: Predicted weather condition




### 1. Clone the repository

git clone https://github.com/thePython2016/weatherPredictionModel.git
cd weatherPredictionModel


### 2. Install dependencies


pip install -r requirements.txt


### 3. Run the app


streamlit run app.py


## Sample Data

Use the provided `sample_data.csv` to test the app format.

| precipitation | temp_max | temp_min | wind |
| ------------- | -------- | -------- | ---- |
| 0.0           | 12.8     | 5.0      | 4.7  |
| 10.9          | 10.6     | 2.8      | 4.5  |

open source and available under the MIT License.
