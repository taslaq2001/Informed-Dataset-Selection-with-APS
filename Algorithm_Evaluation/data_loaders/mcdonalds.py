import zipfile
import pandas as pd

from .loader import Loader


class mcdonalds (Loader) :

    @staticmethod
    def load_from_file(source_path, user_column_name, item_column_name, rating_column_name, timestamp_column_name,
                       **additional_parameters):
        file_path = f"{source_path}/McDonald_s_Reviews.csv"


        df = pd.read_csv(file_path)


        df = df.rename(columns={
            'reviewer_id':user_column_name,
            'product_item': item_column_name,
            'the_Review': rating_column_name,          
            'Review_Date': timestamp_column_name
        })
        df[rating_column_name] = df[rating_column_name].str.extract(r'(\d+)').astype(float)
        df = df.dropna(subset=[rating_column_name, timestamp_column_name])

        return df[[ user_column_name,item_column_name, rating_column_name,timestamp_column_name]]
