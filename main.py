"""CSC110 Fall 2020 Final Project

=================================
This module runs the visualization of the data. There is an option to also run the
cleaning and regressions before visualization, however note they take a very long time
roughly 1 hr together.
"""

import pandas as pd
import numpy as np
import cartopy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import List, Tuple
import os.path # this is imported for a precondition

import clean_data
import regression

LONGITUDE_LIST = [-180, -120, -60, 0, 60, 120, 180]
LATITUDE_LIST = [-90, -60, -30, 0, 30, 60, 90]


def initialize_data(path='Temps4.csv') -> List[pd.DataFrame]:
    """Reads in data from .csv file, slices based on whenever year changes, and returns
    sliced data.

    Preconditions:
      - os.path.isfile(path)
    """
    data = pd.read_csv(path)
    data.sort_values(['year', 'lat', 'lon'], inplace=True)
    data['Change'] = data['year'].diff()
    rows_i = data.index[data['Change'] == 1.0].insert(0, 0)
    data.drop('Change', 1)

    sliced_data = []
    for i in range(len(rows_i) - 1):
        sliced_data.append(data[rows_i[i]: rows_i[i + 1]])

    return sliced_data


def transform_data(sliced_data: List[pd.DataFrame], year: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given the list of dataframes by year, returns a single year of data in the form:

         x1    |    x2   |    x3
    y1 T(1, 1) | T(2, 1) | T(3, 1)
    y2 T(1, 2) | T(2, 2) | T(3, 2)
    y3 T(1, 3) | T(2, 3) | T(3, 3)

    Where T is the temp anomaly of the corresponding coordinate.

    Note: the return is not a table like the one above, instead it is three numpy arrays
    corresponding to the lon, lat, and temp anomaly. The table helps visualize it.

    Preconditions:
      - len(sliced_data) > 0
      -  0 <= year - 1880 <= len(sliced_data)
    """
    index = int(year) - 1880
    single_year = sliced_data[index]
    x_cor = np.unique(single_year['lon'])
    y_cor = np.unique(single_year['lat'])
    lon2d, lat2d = np.meshgrid(x_cor, y_cor)
    temp_anomaly = np.array(single_year.pivot(index='lat', columns='lon', values='temp anomaly'))

    return lon2d, lat2d, temp_anomaly


def generate_map(lon2d: np.ndarray, lat2d: np.ndarray, temp_anomaly: np.ndarray) -> None:
    """Generates a map using contourf based on given data.
    """

    plt.figure(num='Map', figsize=(16, 9))
    global heatmap
    heatmap = plt.axes(projection=cartopy.crs.PlateCarree())
    heatmap.set_global()
    heatmap.coastlines()
    heatmap.gridlines()
    heatmap.set_xticks(LONGITUDE_LIST)
    heatmap.set_yticks(LATITUDE_LIST)
    image = heatmap.contourf(lon2d, lat2d, temp_anomaly, levels=np.linspace(-7, 10, 11), cmap='jet', vmin=-7, vmax=10)
    image.set_clim(-7, 10)
    bar = plt.colorbar(image, fraction=0.0235, pad=0.04)
    bar.set_label('Temp Anomaly', rotation=270)


def update_map(val: int) -> None:
    """Updates map based on given year.

    Preconditions:
      - val >= 1880"""
    year = val
    lon2d, lat2d, temp_anomaly = transform_data(sliced, year)
    heatmap.clear()
    heatmap.set_global()
    heatmap.coastlines()
    heatmap.gridlines()
    heatmap.set_xticks(LONGITUDE_LIST)
    heatmap.set_yticks(LATITUDE_LIST)

    heatmap.contourf(lon2d, lat2d, temp_anomaly, levels=10, cmap='jet', vmin=-7, vmax=10)


def full_run(clean=False, predict=False):
    """Choose whether clean and regressions should be run.

    Preconditions:
      - not clean or os.path.isfile('Temps.nc')
      - not predict or os.path.isfile('Temps3.csv')
      """

    if clean:
        clean_data.full_clean(run=True)

    if predict:
        regression.full_regression(run=True)


if __name__ == '__main__':
    # change these to true if want to run entire program
    # Note: clean=True requires the file 'Temps.nc' to be present and
    #       predict=True requires 'Temps3.csv' to be present.
    full_run(clean=False, predict=False)

    sliced = initialize_data()
    lon, lat, temp = transform_data(sliced, 1880)
    generate_map(lon, lat, temp)
    slider = Slider(plt.axes([0.16, 0.05, 0.65, 0.05]), 'Year', valmin=1880, valmax=2035, valstep=1, valinit=1880)
    slider.on_changed(update_map)
