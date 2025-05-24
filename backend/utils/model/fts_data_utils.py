# fts_data_utils.py
import os
import numpy as np
import pandas as pd

def pad_with_last_row(series, fixed_rows):
    series = np.array(series)
    current_rows, columns = series.shape
    if current_rows > fixed_rows:
        return series[:fixed_rows, :]
    elif current_rows < fixed_rows:
        last_row = series[-1, :]
        padding = np.tile(last_row, (fixed_rows - current_rows, 1))
        return np.vstack((series, padding))
    return series

def load_segmented_time_series(input_dir, max_len=300):
    x_data, y_data = [], []
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv') and file_name != 'multivariate_time_series.csv':
            class_label = file_name.split('_')[0]
            df = pd.read_csv(os.path.join(input_dir, file_name))
            time_series = df.iloc[:, 3:].values
            time_series = pad_with_last_row(time_series, max_len)
            x_data.append(time_series)
            y_data.append(class_label)
    return np.array(x_data), np.array(y_data)