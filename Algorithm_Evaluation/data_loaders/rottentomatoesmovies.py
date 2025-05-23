import zipfile
import pandas as pd

from .loader import Loader


class rottentomatoesmovies (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/Rotten Tomatoes Movies.csv"

        df = pd.read_csv(file_path, on_bad_lines='skip')
        import numpy as np
        df['user'] = np.random.randint(1, 101, size=len(df))  # 100 fake 


        df = df.rename(columns={
            'user':user_column_name,
            'movie_title': item_column_name,
            'review': rating_column_name,          
            'on_streaming_date': timestamp_column_name,
            'genre':'feature Str'
        })
        df = df.loc[:, ~df.columns.duplicated()]

        #df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature Str']]
