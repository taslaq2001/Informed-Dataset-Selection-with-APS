import zipfile
import pandas as pd

from .loader import Loader


class hashedwabreviews (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/hashed_wab_reviews.csv"

        df = pd.read_csv(file_path)

        df = df.rename(columns={
            'userName':user_column_name,
            'appId': item_column_name,
            'score': rating_column_name,          
            'date': timestamp_column_name
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name]]