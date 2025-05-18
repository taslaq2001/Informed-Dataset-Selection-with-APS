import zipfile
import pandas as pd

from .loader import Loader


class bookssecond (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/data.csv"

        df = pd.read_csv(file_path)
        import numpy as np
        df['user'] = np.random.randint(1, 101, size=len(df))  # 100 fake 

        df = df.rename(columns={
            'user':user_column_name,
            'title': item_column_name,
            'average_rating': rating_column_name,          
            'published_year': timestamp_column_name,
            'categories': 'feature2 Str',
            'authors': 'feature1 Str'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature2 Str','feature1 Str']]