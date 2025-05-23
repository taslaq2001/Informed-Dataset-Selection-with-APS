import zipfile
import pandas as pd

from .loader import Loader


class hotels (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/Hotel_Reviews.csv"


        df = pd.read_csv(file_path)
        import numpy as np
        df['user'] = np.random.randint(1, 10000000, size=len(df))  # 100 fake 

        df = df.rename(columns={
            'user':user_column_name,
            'product_item': item_column_name,
            'the_Review': rating_column_name,          
            'Review_Date': timestamp_column_name,
            'feature': 'feature Str'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature Str']]
