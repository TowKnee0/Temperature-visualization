# Temperature-visualization

## Overview

This program processes and plots temperature data onto a world map. The `clean_data.py` file is solely for processing data retrieved from [NASA GISTEMP gridded temperature anomaly data](https://data.giss.nasa.gov/gistemp/). `main.py` can visualize any data from a csv file with 'lat', 'lon', 'year', 'temp anomaly' as column names. `regression.py` performs an 
exponential regression on the data and merges the new data with the old.
