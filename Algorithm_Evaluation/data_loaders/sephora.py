import zipfile
import pandas as pd

from .loader import Loader


class sephora (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/reviews_1250-end_masked.csv"


        df = pd.read_csv(file_path)


        df = df.rename(columns={
            'Unnamed':user_column_name,
            'product_item': item_column_name,
            'the_Review': rating_column_name,          
            'Review_Date': timestamp_column_name,
            'feature': 'feature Str',
            'feature_second':'feature_second'
        })

        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name,'feature Str','feature_second']]
