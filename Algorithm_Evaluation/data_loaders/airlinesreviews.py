import zipfile
import pandas as pd

from .loader import Loader


class airlinesreviews (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/airlines_reviews.csv"

        df = pd.read_csv(file_path)

        df = df.rename(columns={
            'Name':user_column_name,
            'Airline': item_column_name,
            'Overall Rating': rating_column_name,          
            'Review Date': timestamp_column_name,
            'Class': 'feature'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature']]