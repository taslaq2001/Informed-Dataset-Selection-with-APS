import zipfile
import pandas as pd

from .loader import Loader


class adidasvsnike (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/Adidas Vs Nike.csv"


        df = pd.read_csv(file_path)
        #import numpy as np
        #df['user'] = np.random.randint(1, 101, size=len(df))  # 100 fake users
        df = df.rename(columns={
            'Product Name':user_column_name,
            'Product ID': item_column_name,
            'Rating': rating_column_name,          
            'Last Visited': timestamp_column_name,
            'Sale Price': 'feature'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature']]
