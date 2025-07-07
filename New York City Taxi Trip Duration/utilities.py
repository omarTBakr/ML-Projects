'''
    this file will contain simple but useful utilities we will use to
    1. load data
    2. do some preprocessing and transformation over that data
'''


import pandas as pd
import shapely.wkt
import numpy as np
import sys
from constants import train_data_path, val_data_path, boro_data_path, traffic_dat_path, weather_data_path, test_data_path



def load_data_frame(path):
    """

    :param path: the path to what ever csv file you want to load
    :return: data frame contains the data
    """
    data = pd.DataFrame(pd.read_csv(path))
    data.reset_index(drop=True, inplace=True)
    return data


def load_train():
    return load_data_frame(train_data_path)


def load_val():
    return load_data_frame(val_data_path)


def load_test():
    return load_data_frame(test_data_path)


def get_boro_df():
    """this function will load the boro data path_boro
    and case `the_geom` object to be of shapely.multipolygon.MultiPolygon object
    """

    boro_df = load_data_frame(boro_data_path)
    boro_df["the_geom"] = boro_df.the_geom.apply(lambda x: shapely.wkt.loads(x))
    return boro_df


def get_bor_neighbor_avg_dict():
    traffic_df = load_data_frame(traffic_dat_path)
    boro_neighbor_avg_dict = {}

    for neighbor in traffic_df.Boro.unique():  # get unique boro(s) the traffic dataset
        df = traffic_df[traffic_df.Boro == neighbor]  # get the data reated to a singel boro
        boro_neighbor_avg_dict[neighbor] = np.average(df.Vol)  # get the average of that boro
    boro_neighbor_avg_dict["unknown"] = np.average(
        tuple(boro_neighbor_avg_dict.values())
    )  # map any unknown area to 'unknown' which is just the avearage of all other boros
    return boro_neighbor_avg_dict


def get_weather_df():
    """
    this function will load the weather data into a pandas data frame
    reset the index to be the dat
    do some pre-porcessing to convert any string into a numeric
    and return it
    """
    weather_df = load_data_frame(weather_data_path)
    weather_df.set_index("date", inplace=True)
    weather_df.precipitation = weather_df.precipitation.apply(lambda x: float(x) if x != "T" else 0.001)
    weather_df["snow fall"] = weather_df["snow fall"].apply(lambda x: float(x) if x != "T" else 0.001)
    weather_df["snow depth"] = weather_df["snow depth"].apply(lambda x: float(x) if x != "T" else 0.001)
    return weather_df


def transform_split(data: pd.DataFrame):
    from featureEngineering import CustomDataTransformation

    print("=======Data Before Transformation ==========", data.shape)
    print("=====================Transforming Data =================")
    transformer = CustomDataTransformation()
    x_tranformed, y_transformed = transformer.transform(data)
    print("=======Data After Transformation ==========", x_tranformed.shape)
    print("=====================Finished Transforming  =============")

    return x_tranformed, y_transformed
