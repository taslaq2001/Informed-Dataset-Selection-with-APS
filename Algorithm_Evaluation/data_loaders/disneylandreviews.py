import zipfile
import pandas as pd

from .loader import Loader


class disneylandreviews (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/DisneylandReviews.csv"

        df = pd.read_csv(file_path, on_bad_lines='skip', encoding='latin1')
        import numpy as np
        df['user'] = np.random.randint(1, 101, size=len(df))  # 100 fake 

        df = df.rename(columns={
            'user':user_column_name,
            'Branch': item_column_name,
            'Rating': rating_column_name,          
            'Year_Month': timestamp_column_name
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name]]