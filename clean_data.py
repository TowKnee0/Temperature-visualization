"""CSC110 Fall 2020 Final Project

=================================
This module had functions that perform the transforming of data from NetCDF to CSV,
and cleaning the CSV data to be suitable for visualization.
"""

import csv
from typing import List, Tuple
import pandas as pd
import xarray as xr


def convert_to_csv(path='Temps.nc'):
    """Converts NetCDF format into csv file.

    WARNING: This takes a long time. Roughly 15 minutes.
    """

    netcdf_file_name = path
    netcdf_file_in = netcdf_file_name
    csv_file_out = netcdf_file_name[:-3] + '.csv'
    ds = xr.open_dataset(netcdf_file_in)
    df = ds.to_dataframe()

    df.to_csv(csv_file_out)


def read_file(path: str) -> Tuple[str, List[str]]:
    """Returns a tuple of header and data of a given .csv file at a given path."""
    with open(path, 'r') as reader:
        header = next(reader)
        data = [row for row in reader]
    return (header, data)


def clean_data_1(path='Temps.csv') -> None:
    """Take out data with no temp anomaly by mutation.

    WARNING: This takes a long time. Roughly 10 minutes.
    """
    data = pd.read_csv(path)
    data.dropna(subset=['tempanomaly'], inplace=True)
    data.to_csv('Temps1.csv', index=False)


def clean_data_2(path='Temps1.csv') -> None:
    """Group data from the same year and location averaging out temp anomaly.

    This is done by writing to a file because the ram required for returning another list would
    be too high.
    """
    data = read_file(path)
    lat = data[1][0].rstrip().split(',')[0]
    lon = data[1][0].rstrip().split(',')[1]
    year = data[1][0].rstrip().split(',')[3][:4]
    avg = []
    with open('Temps2.csv', 'w', newline='') as f:
        write = csv.writer(f)
        write.writerow(['lat', 'lon', 'year', 'temp anomaly'])
        for row in data[1]:
            temp = row.rstrip().split(',')
            if temp[0] != lat or temp[1] != lon or temp[3][:4] != year:
                write.writerow([lat, lon, year, sum(avg) / len(avg)])
                avg.clear()
                lat, lon, year = temp[0], temp[1], temp[3][:4]
            avg.append(float(temp[-1]))


def clean_data_3(path='Temps2.csv') -> None:
    """Sorts data by year, latitude, longitude in decreasing priority. Also removes duplicate rows then
    outputs into a new .csv file.
    """

    data = pd.read_csv(path)
    data.sort_values(['year', 'lat', 'lon'], inplace=True)
    data.drop_duplicates(inplace=True)
    data.to_csv('Temps3.csv', index=False)


def full_clean(run=False) -> None:
    """A full clean from the NetCDF file to the cleaned csv file.
    """
    if run:
        convert_to_csv()
        clean_data_1()
        clean_data_2()
        clean_data_3()


if __name__ == '__main__':
    # change to True if want to run
    full_clean(run=False)
