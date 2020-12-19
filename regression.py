"""CSC110 Fall 2020 Final Project

==================================
This module contains the functions needed to perform exponential regressions and predictions
on data from Temps3.csv

The new predicted data can be merged with the old recorded data and saved into a file
named Temps4.csv
"""

import pandas as pd
import scipy.optimize
import numpy as np
import os.path
from typing import List, Tuple

def exp_func(x: float, a: float, b: float, c: float) -> float:
    """Evaluate exponential expression.

    Preconditions:
      - b != 0 or x != 0
      """
    return a * (b ** x) + c


def regression(func: callable, xdata: pd.Series, ydata: pd.Series, max: int, p0: List[float]) -> np.array:
    """Performs a regression based on given function with xdata and ydata. Returns the variables
    that best fit the given function.

    Preconditions:
      - len(xdata) > 1 and len(ydata) > 1
      - max > 1
    """

    popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata, p0=p0, maxfev=max)
    return popt


def get_variables(years=100, start=0, filepath='Temps3.csv') -> List[Tuple[np.array, Tuple[float, float]]]:
    """Calculates and returns variables of best fit for each coordinate based on data from
    the latest {years} years.

    Preconditions:
      -  1 < years < 141
      - os.path.isfile(filepath)
      - 0 <= start < years
    """

    data = pd.read_csv(filepath)
    data.sort_values(['lat', 'lon', 'year'], inplace=True, ignore_index=True)

    # subtract 1880 because data starts at year 1880
    # find and record when coordinates change
    data['year'] -= 1880
    data['Change'] = abs(data['lat'].diff()) + abs(data['lon'].diff())
    rows_i = data.index[data['Change'] != 0]

    # slice dataframe based on when coordinates change
    sliced_data = []
    for i in range(0, len(rows_i) - 1):
        sliced_data.append(data[rows_i[i]: rows_i[i + 1]])

    # in case RuntimeError occurred, no need to start from beginning
    sliced_data = sliced_data[start:]

    # variables: stores the variables of best fit for each coordinate
    # counter: keeps track of progress so if RuntimeError occurred, know where to start from
    variables = []
    counter = 0
    for coor in sliced_data:
        # reports progress of calculation since it takes a while
        if counter % 100 == 0:
            print(f'{counter / len(sliced_data) * 100} %')

        # Keep only the latest {years} years
        temp_x = coor['year']
        temp_y = coor['temp anomaly']
        temp_x.reset_index(drop=True, inplace=True)
        temp_y.reset_index(drop=True, inplace=True)

        x = temp_x.iloc[len(temp_x.index) - years:]
        y = temp_y.iloc[len(temp_y.index) - years:]

        if len(x) >= 10:  # not accurate enough if coordinate has less than 10 points
            try:
                variables.append((regression(exp_func, x, y, 150000, p0=[3560, 1, -3500]),
                                  (coor['lon'].iloc[0], coor['lat'].iloc[0])))

            # If the regression could not find best fit in 150000 runs, either increase the number of runs, or
            # inspect the data so far and change its p0.
            # This exception saves the current data into a NPY file and returns the counter so can start from
            # where it left off.
            except RuntimeError:
                if not os.path.isfile(f'variables{years}_years.npy'):
                    np.save(f'variables{years}_years.npy', variables, allow_pickle=True)

                else:
                    temp = list(np.load(f'variables{years}_years.npy', allow_pickle=True))
                    temp.extend(variables)
                    np.save(f'variables{years}_years.npy', temp, allow_pickle=True)

                print(f'Exception on index {counter}')
                break
        counter += 1

    return variables


def predict_data(variables: List[Tuple[np.array, Tuple[float, float]]], years=20) -> pd.DataFrame:
    """Predict data for the next {years} years based on given variables of best fit for each coordinate.
    Returns a pandas dataframe with this new data.

    Preconditions:
      - len(variables) > 1
      - years > 0
    """

    new_data = []

    for data in variables:
        for year in range(1, years + 1):
            # add year + 140 since 2020 - 1880 = 140 and this is predicting the new years (after 2020)
            new_data.append([data[1][1], data[1][0], year + 140,
                             exp_func(year + 140, data[0][0], data[0][1], data[0][2])])

    new_dataframe = pd.DataFrame(new_data, columns=['lat', 'lon', 'year', 'temp anomaly'])

    # add back the 1880 to get true year
    new_dataframe['year'] += 1880

    return new_dataframe


def merge_data(predicted_data: pd.DataFrame, old_data_path='Temps3.csv'):
    """Merges dataframe with existing data in .csv file. Writes combined data
    into a new .csv file.

    Precondition:
      - os.path.isfile(old_data_path)
    """
    old = pd.read_csv(old_data_path)
    merged = pd.concat([old, predicted_data], axis=0)
    merged.sort_values(['year', 'lat', 'lon'], inplace=True)
    merged.to_csv('Temps4.csv', index=False)


def full_regression(run=False) -> None:
    """Performs the entire exponential regression and merging.
    """
    if run:
        variables = get_variables()
        new = predict_data(variables)
        merge_data(new)


if __name__ == '__main__':
    # change to True to run
    full_regression(run=False)
