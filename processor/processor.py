import pandas as pd
import numpy as np
import bson

class Processor:
    def __init__(self):
        pass
    
    def load_multiple_bson_objects(self, file_path):
        with open(file_path, 'rb') as file:
            return [document for document in bson.decode_all(file.read())]
    
    def convert_to_data_frame(self, row_data, keyboard_data=None):
        data_frames = []
        for data_point in row_data:
            current_user = data_point['user_id']
            data_to_convert = data_point['events'] if keyboard_data else data_point['data']
            current_data_frame = pd.DataFrame(data_to_convert)
            current_data_frame['user_id'] = current_user
            data_frames.append(current_data_frame)

        df = pd.concat(data_frames, ignore_index=True)
        df = df.drop_duplicates()
        return df
    
    def add_lagged_features(self, df, lagged_features):
        for feature in lagged_features:
            df[f'{feature}_lag1'] = df[feature].shift(1)
        return df
    
    def add_rolling_statistics(self, df, rolling_features, window=5):
        for feature in rolling_features:
            df[f'rolling_mean_{feature}'] = df[feature].rolling(window=window).mean()
        return df
    
    def seasonal_decomposition(self, df, seasonal_features, window=24*60):
        for feature in seasonal_features:
            df[f'trend_{feature}'] = df[feature].rolling(window=window, center=True, min_periods=1).mean()
            df[f'seasonal_{feature}'] = df[feature] - df[f'trend_{feature}']
            df[f'residual_{feature}'] = df[feature] - df[f'trend_{feature}'] - df[f'seasonal_{feature}']
        return df
    
    def extract_time_features(self, df):
        time_features = ['hour', 'day_of_week', 'second', 'millisecond']
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['second'] = df.index.second
        df['millisecond'] = df.index.microsecond // 1000
        for feature in time_features:
            df[f'{feature}_sin'] = np.sin(2 * np.pi * df[feature] / df[feature].max())
            df[f'{feature}_cos'] = np.cos(2 * np.pi * df[feature] / df[feature].max())
        df.drop(columns=time_features, inplace=True)
        return df
    
    def build_sequences(self, df, sequence_length):
        sequences = []
        targets = []
        for _, group in df.groupby('user_id'):
            user_data = group.drop('user_id', axis=1).values
            user_targets = group['user_id'].iloc[0]  # Get the target for this group
            for i in range(len(user_data) - sequence_length + 1):
                sequences.append(user_data[i:i+sequence_length])
                targets.append(user_targets)  # Append the same target for each sequence in this group
        return np.array(sequences), np.array(targets)
