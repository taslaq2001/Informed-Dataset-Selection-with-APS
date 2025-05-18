import zipfile
import pandas as pd

from .loader import Loader


class ryanairreviews (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/ryanair_reviews.csv"

        df = pd.read_csv(file_path)
        import numpy as np
        df['user'] = np.random.randint(1, 101, size=len(df))  # 100 fake 

        df = df.rename(columns={
            'user':user_column_name,
            'Value For Money': item_column_name,
            'Overall Rating': rating_column_name,          
            'Date Published': timestamp_column_name,
            'Seat Type': 'feature Str'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature Str']]