'''
this file will contain pathes to the data we need like
1. weather data
2. traffic data
3. boro data
4. train , val , test data (you wanna test the model on)
'''

import os

# split sample
# train_data_path = os.path.join("split", "split_sample", "train.csv")
# val_data_path = os.path.join("split", "split_sample", "test.csv")


# full kagge data
train_data_path = os.path.join("split", "train.csv")
val_data_path = os.path.join("split", "val.csv")

# comment the above and uncomment those to test and evaluate over kaggle data
# train_data_path = os.path.join('Data_From_kaggle','train.csv')
# val_data_path = os.path.join('Data_From_kaggle','test.csv')

test_data_path = os.path.join("split", "test.csv")

boro_data_path = os.path.join("Borough_Data", "Borough_Boundaries.csv")
traffic_dat_path = os.path.join("Traffic_Data", "Traffic_Volume_Counts_2016.csv")
weather_data_path = os.path.join("weather data", "weather_data_nyc_centralpark_2016(1).csv")
