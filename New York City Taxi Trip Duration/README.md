

---

# New York Taxi Trip Duration

In this project, we will examine the "New York City Taxi Trip Duration" dataset. The original dataset and competition details can be found on   Kaggle [here](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)

## Data Fields

The dataset includes the following fields:

*   **`id`**: A unique identifier for each trip.
*   **`vendor_id`**: A code indicating the provider associated with the trip record.
*   **`pickup_datetime`**: Date and time when the meter was engaged.
*   **`dropoff_datetime`**: Date and time when the meter was disengaged (Note: This column was dropped in our analysis).
*   **`passenger_count`**: The number of passengers in the vehicle (driver-entered value).
*   **`pickup_longitude`**: The longitude where the meter was engaged.
*   **`pickup_latitude`**: The latitude where the meter was engaged.
*   **`dropoff_longitude`**: The longitude where the meter was disengaged.
*   **`dropoff_latitude`**: The latitude where the meter was disengaged.
*   **`store_and_fwd_flag`**: This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server (Y = store and forward; N = not a store and forward trip).
*   **`trip_duration`**: Duration of the trip in seconds (this is the target variable).

The original data is from a contest held around 2016.

## Important Considerations

*   We will not focus on model regularization, workarounds, or trying many different models (as this is often a mechanical process). Instead, our primary focus will be on **feature engineering**.
*   Several external datasets are used in conjunction with the primary dataset. Ensure these are downloaded and set up correctly:
    1.  **Automated Traffic Volume Counts:** Provided by the New York City Department of Transportation. Can be found [here](https://catalog.data.gov/dataset/automated-traffic-volume-counts#:~:text=New%20York%20City%20Department%20of,vary%20from%20year%20to%20year).
    2.  **Weather Data in New York City - 2016:** Available on Kaggle [here](https://www.kaggle.com/datasets/mathijs/weather-data-in-new-york-city-2016).
    3.  **Borough Boundaries for New York City:** Can be found [here](https://catalog.data.gov/dataset/borough-boundaries-a8eb4#:~:text=Metadata%20Updated%3A%20April%2012%2C%202025).
    4.  **New York OpenStreetMap Data from Geofabrik:** The `osm.pbf` file can be downloaded from [here](https://download.geofabrik.de/north-america/us/new-york.html).
        *   Querying this file with start (longitude, latitude) and end (longitude, latitude) locations will return the estimated travel time in seconds based on OpenStreetMap data.
        *   A Docker image needs to be built for this file to enable querying via an OSRM (Open Source Routing Machine) server.
        *   _All instructions for setting up the Docker container can be found in `docker_setup.md`._
*   **If you choose not to download and set up the OSRM data (Geofabrik file):**
    *   You can still run the Exploratory Data Analysis (EDA) but will need to skip OSRM-dependent features.
    *   For the provided code, comment out the following line in `featureEngineering.py`:
        ```python
        # self.osrm,
        ```
*   A `requirements.txt` file is included. Ensure you install the dependencies using `pip install -r requirements.txt` before running the EDA or other scripts.

## Attached Files and Folders

Below is an overview of the project's file structure:
```
.
├── Borough_Data
│   ├── Borough_Boundaries.csv
│   └── datalink.txt
├── constants.py
├── Data_From_kaggle
│   ├── sample_submission.zip
│   ├── test.csv
│   └── train.csv
├── docker_setpup.md
├── EDA
│   ├── EDA_NewYork_City_Taxi_Trip_Duration.html
│   ├── EDA_NewYork_City_Taxi_Trip_Duration.ipynb
│   ├── EDA_NewYork_City_Taxi_Trip_Duration.pdf
│   └── map.html
├── featureEngineering.py
├── initialExploration.ipynb
├── main.py
├── modeling.py
├── models
│   ├── Ridge_2025-05-23_05-40-15.pkl
│   └── Ridge_2025-05-30_13-03-35.pkl
├── README.md
├── Report
│   ├── LaTex Template for New York Trip Duration Report.zip
│   └── New_York_Trip_Duration_Report.pdf
├── requirements.txt
├── split
│   ├── split_sample
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
│   ├── test.csv
│   ├── train.csv
│   └── val.csv
├── Traffic_Data
│   ├── datalink.txt
│   └── Traffic_Volume_Counts_2016.csv
├── utilities.py
└── weather data
    ├── datalink.txt
    └── weather_data_nyc_centralpark_2016(1).csv

```
**Note:**
*   The HTML version of the EDA (`EDA_NewYork_City_Taxi_Trip_Duration.html`) is included as it contains interactive visualizations that may not be fully available in the PDF version.
*   A LaTeX template for the report is included in case you need it.
*   All necessary data files (or links to them) are expected to be within the appropriately named folders.

## Data Exploration

### Target variable

*   We have taken the log of `trip_duration` (i.e., `log(trip_duration)`) to reduce its variance.
*   Outlier journeys (e.g., trips with distances >1000 km, if applicable, or extremely long/short durations based on EDA) were identified and removed.

### Feature analysis

1.  **Null values and data types:**
    *   The data was generally clean with minimal null values.
    *   Most data types were appropriate, except for `pickup_datetime`, which required casting to a datetime object.
    *   **Note:** We decided to drop the `dropoff_datetime` column to prevent data leakage if used directly as a feature and to calculate duration from pickup time and the target variable.

2.  **Pickup datetime:**
    *   The trips in the dataset are predominantly from January 2016 to June 2016 (inclusive). It's possible that data from other months were reserved for a private test set in the original competition.
    *   Trip distribution is relatively uniform across days of the week and days of the month within this period.

3.  **Vendor ID:**
    *   There are two unique vendor IDs (1 and 2), suggesting two distinct service providers.
    *   The average trip duration is approximately 845.4 seconds for vendor 1 and 1058.64 seconds for vendor 2.
    *   Despite this difference in averages, the overall trip duration distributions for each vendor, both overall and per month, appear quite similar.

4.  **Passenger count:**
    *   A noticeable difference is observed here: vendor 2 records trips with a significantly higher maximum passenger count (up to 6 passengers), while vendor 1 typically has lower passenger counts.

### OSRM API

*   OSRM (Open Source Routing Machine) is a high-performance routing engine used to find shortest paths in road networks.
*   It was used in this project to estimate the travel distance and time between pickup (longitude, latitude) and dropoff (longitude, latitude) coordinates.
*   **Note:** Since the trip data is from 2016, estimates from the OSRM API (which uses more current road network data, e.g., from 2025) might differ from the actual historical travel times and distances. This is an important factor to consider when interpreting these features.

### Geographic information

*   **Borough information:**
    *   Mapping pickup and dropoff locations to New York City boroughs reveals that Manhattan accounts for the largest share of trips among the five boroughs (refer to visualizations like Fig. 7 in the EDA report).
    *   The New York Borough regions data (mentioned in "Important Considerations") was used for this mapping. Each borough is geographically defined by a 'MultiPolygon' shape.

### Traffic volume

*   Based on the "Automated Traffic Volume Counts" dataset, analysis can reveal insights into borough-specific traffic patterns. For instance, initial exploration might suggest Queens as one of the busiest boroughs (refer to visualizations like Fig. 8 in the EDA report).
*   **Note:** An 'unknown' category has been added to assign to trips whose coordinates do not fall within known borough boundaries.

### Weather data

*   The weather data (link provided in "Important Considerations") is relatively clean.
*   All available features (e.g., temperature, precipitation, snowfall, wind speed) were considered, as weather conditions are expected to significantly impact trip duration.

## Modeling

### Data pipeline

The modeling process involves the following key steps:

1.  **Feature splitting:** Separating features from the target variable.
2.  **Custom feature engineering:** Creating new features based on domain knowledge and EDA insights (detailed in `featureEngineering.py`).
3.  **One-Hot Encode categorical features:** Converting categorical variables into a numerical format suitable for machine learning models.
4.  **Polynomial Features (degree=2):** Generating interaction terms and polynomial features to capture non-linear relationships.
5.  **Scaling:** Standardizing features (e.g., using StandardScaler).
6.  **Model Training:** Using algorithms like Ridge Regression.

### Results

The following table summarizes the model performance on the train, validation, and test sets:

| Dataset | Shape Before Transformation | Shape After Transformation | RMSE   | R² Score |
| :------ | :-------------------------- | :------------------------- | :----- | :------- |
| Train   | (1,000,000, 10)             | (980,000, 55)              | 0.3973 | 0.7146   |
| Val     | (229,319, 10)               | (224,731, 55)              | 0.4010 | 0.7107   |
| Test    | (229,322, 10)               | (224,734, 55)              | 0.3969 | 0.7156   |

*(Note: Shapes are (samples, features). RMSE and R² Score are for the log-transformed target variable.)*

## What Could Be Enhanced

*   **Better traffic volume information:** Incorporate more granular or real-time (if available historically) traffic data.
*   **Time/Distance metrics per borough segment:** Extract average 'time/distance' metrics for trips between specific boroughs (e.g., Queens to Manhattan) segmented by time of day (morning, afternoon, evening) and day of the week.
*   **More granular weather information:** Utilize features like wind direction and compare it to trip direction; explore interactions between different weather variables.
*   **Advanced feature interactions:** Explore more explicit feature interactions beyond those captured by polynomial features, possibly using domain expertise or automated methods.
*   **Alternative modeling techniques:** Experiment with gradient boosting models (XGBoost, LightGBM, CatBoost) or neural networks, which often perform well on tabular data.
*   **Hyperparameter optimization:** Conduct more thorough hyperparameter tuning for the chosen models.

---

