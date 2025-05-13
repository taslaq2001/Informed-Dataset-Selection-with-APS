import zipfile
import pandas as pd

from .loader import Loader


class wines_SPA (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/wines_SPA.csv"

        df = pd.read_csv(file_path)

        import numpy as np
        df['user'] = np.random.randint(1, 101, size=len(df))  # 100 fake 

        df = df.rename(columns={
            'user':user_column_name,
            'wine': item_column_name,
            'rating': rating_column_name,          
            'year': timestamp_column_name,
            'price': 'feature'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature']]