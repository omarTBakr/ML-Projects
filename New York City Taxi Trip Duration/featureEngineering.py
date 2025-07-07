"""
 this file will do all the feature engineering to the data and return
 x : numpy.array
 x : numpy.array
 no data leakage (guranteed)
"""

import numpy as np
import pandas as pd
import requests
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
import sys
import holidays
from geopy import distance
import astral
from astral import LocationInfo
from astral.sun import sun
from astral import Observer
from utilities import get_boro_df, get_bor_neighbor_avg_dict, get_weather_df
from tqdm import tqdm




class CustomDataTransformation(BaseEstimator):
    """
    this class will handle all custom feature engineering ,
    processing done over the data
    """

    def __init__(
        self,
        drop_features=[
            "id",
            "vendor_id",
            "store_and_fwd_flag",
            "pickup_datetime",
            "pickup_boro",
            "dropoff_boro",
            "dropoff_datetime",  # this for kaggle data only
        ],
    ):
        super().__init__()
        self.features_to_drop = drop_features

    def fit(self, data, y=None):
        self.data = data
        return self

    # def handle_id(self, data: pd.DataFrame):
    #     """
    #     drop the id feature
    #
    #     :param data:  data frame
    #     :return: data after id dropped
    #     """
    #     data.drop("id", inplace=True, axis=1)
    #     return data
    #
    # def handle_fwd_flag(self, data: pd.DataFrame):
    #     """this function will drop the fwd flag
    #     and return the dataframe
    #     """
    #     data.drop("id", inplace=True, axis=1)
    #     return data

    def handle_date(self, data: pd.DataFrame):
        """this function will transform the 'pcikup_datetime' object into a date and
        create this features
        month : month of the trip
        day: day of the trip
        hour: hour of the tirp
        minute: miniute of the trip
        week: week of the month
        week_day: day of the week (number)
        week_of_year: which week of the year (year has 48 weeks_
        quarter: which quarter of the year (year has 4 quarters)
        weekend_status: (binary indicator)weather this day is a weekend or not ?
        is_holiday: (binary indicator)if this day is a notional holiday or not ?
        which they all depend on the date somehow

        and some cyclic features to help the model indentify the cyclic nature of this feature
        """
        data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
        data["month"] = data["pickup_datetime"].apply(lambda x: x.month)
        data["day"] = data["pickup_datetime"].apply(lambda x: x.day)
        data["hour"] = data["pickup_datetime"].apply(lambda x: x.hour)
        data["minute"] = data["pickup_datetime"].apply(lambda x: x.minute)
        data["week"] = data["pickup_datetime"].apply(lambda x: x.week)
        data["week_day"] = data["pickup_datetime"].apply(lambda x: x.day_of_week)
        data["week_of_year"] = data["pickup_datetime"].apply(lambda x: x.weekofyear)
        data["quarter"] = data["pickup_datetime"].apply(lambda x: x.quarter)
        data["weekend_status"] = data["pickup_datetime"].apply(lambda x: 1 if x.day_name() in ("Saturday", "Sunday") else 0)
        # holiday or not
        us_holidays = holidays.US()

        data["is_holiday"] = data["pickup_datetime"].apply(lambda x: 1 if x.date() in us_holidays else 0)

        # cyclic features
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
        data["minute_sin"] = np.sin(2 * np.pi * data["minute"] / 60)
        data["minute_cos"] = np.cos(2 * np.pi * data["minute"] / 60)
        data["week_sin"] = np.sin(2 * np.pi * data["week"] / 4)
        data["week_cos"] = np.cos(2 * np.pi * data["week"] / 4)
        data["week_of_year_sin"] = np.sin(2 * np.pi * data["week_of_year"] / 48)
        data["week_of_year_cos"] = np.cos(2 * np.pi * data["week_of_year"] / 48)
        return data

    def add_distance(self, data: pd.DataFrame):
        """
        this function will add these features to the data
        1. geo_distance : which is the haversine distance
        2.h_distance : which is the horizontal distance between pick up location and drop off location
        3.v_distance : which is the vertical distance between pick up location and drop off location
        4. passenger_per_geodist: which is the fraction between the passenger_count / geo_dist
        :param X:
        :return:
        """

        def geo_dist(row):
            return distance.distance(
                (row["pickup_latitude"], row["pickup_longitude"]),
                (row["dropoff_latitude"], row["dropoff_longitude"]),
            ).km

        data["geo_dist"] = data.apply(geo_dist, axis=1)
        # passenger per km
        data["passenger_per_geodist"] = data["passenger_count"] / (data["geo_dist"] + 1e-3)

        def h_distance(row):
            # 111.3 is the distance between two longitude lines
            return np.abs(row["pickup_longitude"] - row["dropoff_longitude"]) * 111.3

        def v_distance(row):
            # 111.3 is the distance between two latitude lines
            return np.abs(row["pickup_latitude"] - row["dropoff_latitude"]) * 111

        data["h_distance"] = data.apply(h_distance, axis=1)
        data["v_distance"] = data.apply(v_distance, axis=1)
        return data

    def rush_hour(self, data):
        def is_rush_hour(row):
            return 1 if 6 <= row["pickup_datetime"].hour <= 10 or 13 <= row["pickup_datetime"].hour <= 20 else 0

        data["is_rush_hour"] = data.apply(is_rush_hour, axis=1)
        return data

    def osrm(self, data):
        def osrm_distance_time_estimate(row: pd.DataFrame):
            start = f"{row['pickup_longitude']},{row['pickup_latitude']}"
            end = f"{row['dropoff_longitude']},{row['dropoff_latitude']}"

            url = f"http://localhost:5000/route/v1/driving/{start};{end}?overview=false"
            response = requests.get(url)
            duration = None
            distance = None
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                routes = data.get("routes", [])
                if routes:
                    route = routes[0]
                    distance = route.get("distance", 0) / 1000  # in km
                    duration = route.get("duration", 0)  # in seconds
            return (
                distance if distance else row["geo_dist"],  # if there is no route avialable from osrm pass the geo_dist
                (duration if duration else 959.492 if row["geo_dist"] < 50 else row["geo_dist"] * 30),
                # rough estimate if we don't have a dinstance
            )

        distance_time_estimate = data.apply(osrm_distance_time_estimate, axis=1)
        data["osrm_distance"], data["osrm_time"] = (
            distance_time_estimate.apply(lambda x: x[0]),
            distance_time_estimate.apply(lambda x: x[1]),
        )
        data["osrm_time"] = np.log1p(data["osrm_time"])  # since we will apply np.log1p to the tirp_duration
        data["osrm_implied_speed"] = data["osrm_distance"] / (data["osrm_time"] + 1e-4)
        data["geo_implied_speed"] = data["geo_dist"] / (data["osrm_time"] + 1e-4)
        data["osrm_dist/geo_dist"] = data["osrm_distance"] / (data["geo_dist"] + 1e-4)
        data["passenger_per_osrm_dist"] = data["passenger_count"] / (data["osrm_distance"] + 1e-3)
        return data

    def boro(self, data):
        boro_df = get_boro_df()

        def handle_boro(row, col="pickup"):
            from shapely import Point

            if col == "pickup":
                point = Point(row.pickup_longitude, row.pickup_latitude)
            else:
                point = Point(row.dropoff_longitude, row.dropoff_latitude)

            check = boro_df["the_geom"].apply(lambda x: point.within(x))

            neighbor = "".join(map(str, boro_df[check]["BoroName"].values))
            return neighbor if neighbor else "unknown"

        data["pickup_boro"] = data[["pickup_longitude", "pickup_latitude"]].apply(handle_boro, axis=1)

        data["dropoff_boro"] = data[["dropoff_longitude", "dropoff_latitude"]].apply(handle_boro, args=("dropoff",), axis=1)
        return data

    def traffic_volume(self, data):
        boro_neighbor_avg_dict = get_bor_neighbor_avg_dict()

        def volume(row, col="pickup_boro"):
            # can be enhanced to get the exact volume by the hour if available
            return boro_neighbor_avg_dict[row[col]]

        data["traffic_volume_pickup"] = data.apply(volume, args=("pickup_boro",), axis=1)
        data["traffic_volume_dropoff"] = data.apply(volume, args=("dropoff_boro",), axis=1)

        return data

    def weather(self, data):
        weather_df = get_weather_df()

        def add_weather_data(row, col="maximum temperature"):
            day = row["day"]
            month = row["month"]
            weather_row = weather_df.loc[f"{int(day)}-{int(month)}-2016"]
            return weather_row[col]

        data["max_temp"] = data.apply(add_weather_data, args=("maximum temperature",), axis=1)
        data["min_temp"] = data.apply(add_weather_data, args=("minimum temperature",), axis=1)
        data["avg_temp"] = data.apply(add_weather_data, args=("average temperature",), axis=1)
        data["precipitation"] = data.apply(add_weather_data, args=("precipitation",), axis=1)
        data["snow fall"] = data.apply(add_weather_data, args=("snow fall",), axis=1)
        data["snow depth"] = data.apply(add_weather_data, args=("snow depth",), axis=1)

        return data

    def daylight(self, data):
        def daylight_minutes(row):
            # Extract latitude and longitude from the row
            latitude = row["pickup_latitude"]
            longitude = row["pickup_longitude"]

            # Create an Observer object
            observer = Observer(latitude=latitude, longitude=longitude)

            # date = datetime.date(2016, row['month'], row['day'])
            date = row["pickup_datetime"]
            s = sun(observer, date=date, tzinfo="US/Eastern")

            # Calculate daylight duration
            daylight_duration = s["sunset"] - s["sunrise"]

            # Convert duration to total minutes
            total_minutes = daylight_duration.total_seconds() / 60
            return total_minutes

        data["daylight_minutes"] = data.apply(daylight_minutes, axis=1)
        return data

    def one_hot_encode(self, data):
        categories = [["Brooklyn", "Staten Island", "Manhattan", "Bronx", "Queens", "unknown"]]
        encoder = OneHotEncoder(sparse_output=False, categories=categories, dtype=int)
        pickup = encoder.fit_transform(data["pickup_boro"].to_numpy().reshape(-1, 1))
        pick_up_categories = [category + "pickup" for category in categories[0]]
        pickup_df = pd.DataFrame(pickup, columns=pick_up_categories)
        # print(pickup_df.head())

        # drop off
        dropoff = encoder.transform(data["dropoff_boro"].to_numpy().reshape(-1, 1))
        dropoff_categories = [category + "dropoff" for category in categories[0]]
        dropoff_df = pd.DataFrame(dropoff, columns=dropoff_categories)
        # print("drop off ")
        # print(dropoff_df.head())

        return pd.concat([data, pickup_df, dropoff_df], axis=1)

    def drop_features(self, data):
        """
        this function will double-check if the features you want to remove is within the data
        otherwise it will ignore it
        # i have added it because the data in split/ .. does not have dropoff_datetime while the kaggle data have
        """
        for feature in self.features_to_drop:
            if feature not in data.columns:
                self.features_to_drop.remove(feature)
        data.drop(self.features_to_drop, axis=1, inplace=True)
        return data

    # def split_data(self, data):
    #     x_train, y_train = (
    #         data.drop("trip_duration", axis=1).to_numpy(),
    #         np.log1p(data.trip_duration.to_numpy().reshape(-1, 1)),
    #     )
    #     return x_train, y_train

    def to_numpy(self, data: pd.DataFrame):
        return data.drop("trip_duration", axis=1).to_numpy(), np.log1p(data["trip_duration"].to_numpy().reshape(-1, 1))

    def filter_outliers(self, data):
        """
        remove the outliers form the data based on the geo_dist column
        """
        filtered_data = data[
            (data.geo_dist <= data["geo_dist"].quantile(0.99)) & (data.geo_dist >= data["geo_dist"].quantile(0.01))
        ]

        return filtered_data

    def transform(self, data=None):
        if data is not None:
            self.data = data

        functions = [
            self.handle_date,
            self.add_distance,
            self.rush_hour,
            self.osrm,
            self.boro,
            self.traffic_volume,
            self.weather,
            self.daylight,
            self.one_hot_encode,
            self.drop_features,
            self.filter_outliers,
            self.to_numpy,
        ]
        for function in tqdm(functions):
            # column_before = set(self.data.columns)
            # print('applying ',function.__name__ )

            self.data = function(self.data)
            # if function != self.to_numpy:
            # print('columns after transformation are' ,self.data.columns)
            # columns_after = set(self.data.columns)
            # print('columns added are' , columns_after-column_before)

        # print('final shape is ',self.data[0].shape)
        return self.data

    def fit_transform(self, data):
        return self.transform(data)
