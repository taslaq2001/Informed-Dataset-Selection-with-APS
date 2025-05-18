import zipfile
import pandas as pd

from .loader import Loader


class books (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/books.csv"

        df = pd.read_csv(file_path, on_bad_lines='skip')
        import numpy as np
        df['user'] = np.random.randint(1, 101, size=len(df))  # 100 fake 

        df = df.rename(columns={
            'user':user_column_name,
            'isbn': item_column_name,
            'average_rating': rating_column_name,          
            'publication_date': timestamp_column_name,
            'publisher': 'feature Str'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature Str']]