import importlib
import json
import re
from collections import Counter
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from multiprocessing import Process, Manager


def count_elements(data, column, shared_dict):
    shared_dict[column] = np.unique(data, return_counts=True)


def filter_invalid(data, counts, column, core, shared_dict):
    valid_index = np.where(counts[1] > core)
    valid_ids = counts[0][valid_index]

    data = np.isin(data, valid_ids)

    shared_dict[column] = data


class RecSysDataSet:
    available_data_sets = ["AdidasVsNike","airlines_reviews.csv","Bankrupt","books","Books1","Boston","California_Houses","DisneylandReviews.csv","hashed_wab_reviews.csv","Rotten Tomatoes Movies","eyanair_reviews","wind_dataset","wines_SPA"]

    def __init__(self, data_set_name):
        self.data_set_name = data_set_name

        self.data_set_folder_base_path = Path(f"./data_sets/{self.data_set_name}")
        self.source_files_path = Path(f"./data_sets/{self.data_set_name}/source/files")
        self.link_file_path = Path(f"./data_sets/{self.data_set_name}/source/site/link.txt")

        self.processed_folder_path = Path(f"./data_sets/{self.data_set_name}/processed")
        self.processed_data_path = Path(f"./data_sets/{self.data_set_name}/processed/interactions.csv")
        self.processed_data_log_path = Path(f"./data_sets/{self.data_set_name}/processed/processing_log.txt")
        self.processed_data_metadata_path = Path(f"./data_sets/{self.data_set_name}/processed/metadata.json")

        self.atomic_folder_path = Path(f"./data_sets/{self.data_set_name}/atomic")
        self.atomic_data_path = Path(f"./data_sets/{self.data_set_name}/atomic/{self.data_set_name}.inter")
        self.atomic_data_log_path = Path(f"./data_sets/{self.data_set_name}/atomic/processing_log.txt")
        self.atomic_data_metadata_path = Path(f"./data_sets/{self.data_set_name}/atomic/metadata.json")

        self.data = None
        self.data_origin = None
        self.meta_data = None
        self.data_splits = None
        self.data_split_type = None
        self.feedback_type = None

        self.user_column_name = "user"
        self.item_column_name = "item"
        self.rating_column_name = "rating"
        self.timestamp_column_name = "timestamp"

        self.log = []

    # region Path and File Management

    def processed_folder_exists(self):
        return True if self.processed_folder_path.is_dir() else False

    def processed_data_exists(self):
        return True if self.processed_data_path.is_file() else False

    def processed_data_log_exists(self):
        return True if self.processed_data_log_path.is_file() else False

    def processed_data_metadata_exists(self):
        return True if self.processed_data_metadata_path.is_file() else False

    def atomic_folder_exists(self):
        return True if self.atomic_folder_path.is_dir() else False

    def atomic_data_exists(self):
        return True if self.atomic_data_path.is_file() else False

    def atomic_data_log_exists(self):
        return True if self.atomic_data_log_path.is_file() else False

    def atomic_data_metadata_exists(self):
        return True if self.atomic_data_metadata_path.is_file() else False

    def check_data_loaded(self):
        if self.data is not None:
            self.log_and_print(f"Data set already loaded.")
            return True
        else:
            self.log_and_print(f"No data set already loaded.")
            return False

    def write_data(self, force_write=False):
        if not self.check_data_loaded():
            return
        if self.data_origin == "processed":
            if self.processed_data_exists():
                self.log_and_print(f"Processed data set already exists.")
                if force_write:
                    self.log_and_print(f"Overwriting processed data set.")
                else:
                    self.log_and_print(f"Skipping writing processed data set.")
                    return
            self.log_and_print(f"Writing processed data set to file.")
            self.data.to_csv(self.processed_data_path, index=False)
            self.log_and_print(f"Written processed data set to file.")
        elif self.data_origin == "atomic":
            if self.atomic_data_exists():
                self.log_and_print(f"Atomic data set already exists.")
                if force_write:
                    self.log_and_print(f"Overwriting atomic data set.")
                else:
                    self.log_and_print(f"Skipping writing atomic data set.")
                    return
            self.log_and_print(f"Writing atomic data set to file.")
            self.data.to_csv(self.atomic_data_path, index=False)
            self.log_and_print(f"Written atomic data set to file.")
        else:
            self.log_and_print(f"Wrong data origin. Unable to write data.")
            return

    def load_data(self, force_load=False):
        if self.data_origin == "raw":
            if self.check_data_loaded():
                if force_load:
                    self.log_and_print("Unloading data set.")
                    self.unload_data()
                else:
                    self.log_and_print("Skipping raw data loading.")
                    return
            self.log_and_print(f"Loading raw data set.")
            class_name = re.sub(r'[^a-zA-Z0-9]', '', self.data_set_name)
            module_name = f"data_loaders.{re.sub(r'[^a-zA-Z0-9]', '_', self.data_set_name).lower()}"
            try:
                loader = importlib.import_module(module_name).__getattribute__(class_name)
                self.data = loader.load_from_file(source_path=self.source_files_path,
                                                  user_column_name=self.user_column_name,
                                                  item_column_name=self.item_column_name,
                                                  rating_column_name=self.rating_column_name,
                                                  timestamp_column_name=self.timestamp_column_name)
            except ModuleNotFoundError:
                self.log_and_print(f"Module {module_name} for data set {self.data_set_name} not found.")
                self.log_and_print("Skipping raw data loading.")
                return
            self.log_and_print(f"Loaded raw data set.")
            self.set_feedback_type()
            self.data_origin = "raw"
        elif self.data_origin == "processed":
            if self.processed_data_exists():
                if self.check_data_loaded():
                    if force_load:
                        self.log_and_print("Unloading data set.")
                        self.unload_data()
                    else:
                        self.log_and_print("Skipping processed data loading.")
                        return
                self.log_and_print(f"Loading processed data set.")
                self.data = pd.read_csv(self.processed_data_path)
                self.log_and_print(f"Loaded processed data set.")
                self.set_feedback_type()
                self.data_origin = "processed"
        elif self.data_origin == "atomic":
            if self.atomic_data_exists():
                if self.check_data_loaded():
                    if force_load:
                        self.log_and_print("Unloading data set.")
                        self.unload_data()
                    else:
                        self.log_and_print("Skipping atomic data loading.")
                        return
                self.log_and_print(f"Loading atomic data set.")
                self.data = pd.read_csv(self.atomic_data_path)
                self.log_and_print(f"Loaded atomic data set.")
                self.set_feedback_type()
                self.user_column_name = "user_id:token"
                self.item_column_name = "item_id:token"
                self.rating_column_name = "rating:float"
                self.timestamp_column_name = None
                self.data_origin = "atomic"
        else:
            self.log_and_print(f"Processed data set does not exist.")
            return None

    def check_data_splits_exist(self):
        if self.data_origin == "processed":
            for file in self.processed_folder_path.iterdir():
                if "split" in file.name:
                    self.log_and_print(f"Data splits exist.")
                    return True
                else:
                    self.log_and_print(f"Data splits do not exist.")
                    return False
        elif self.data_origin == "atomic":
            for file in self.atomic_folder_path.iterdir():
                if "split" in file.name:
                    self.log_and_print(f"Data splits exist.")
                    return True
                else:
                    self.log_and_print(f"Data splits do not exist.")
                    return False

    def check_data_splits_loaded(self):
        if self.data_splits is not None:
            self.log_and_print(f"Data splits already loaded.")
            return True
        else:
            self.log_and_print(f"No data splits already loaded.")
            return False

    def write_data_splits(self, force_write=False):
        if self.data_origin == "processed":
            if self.check_data_splits_exist():
                self.log_and_print(f"Data splits already exist.")
                if force_write:
                    self.log_and_print(f"Overwriting data splits.")
                else:
                    self.log_and_print(f"Skipping writing data splits.")
                    return
            self.log_and_print(f"Writing data splits to file.")
            if self.data_split_type == "random_ho" or self.data_split_type == "user_ho":
                self.data_splits["train"].to_csv(f"{self.processed_folder_path}/train_split.csv", index=False)
                self.data_splits["valid"].to_csv(f"{self.processed_folder_path}/valid_split.csv", index=False)
                self.data_splits["test"].to_csv(f"{self.processed_folder_path}/test_split.csv", index=False)
            elif self.data_split_type == "random_cv" or self.data_split_type == "user_cv":
                for fold in self.data_splits:
                    self.data_splits[fold]["train"].to_csv(f"{self.processed_folder_path}/train_split_{fold}.csv",
                                                           index=False)
                    self.data_splits[fold]["valid"].to_csv(f"{self.processed_folder_path}/valid_split_{fold}.csv",
                                                           index=False)
                    self.data_splits[fold]["test"].to_csv(f"{self.processed_folder_path}/test_split_{fold}.csv",
                                                          index=False)
        elif self.data_origin == "atomic":
            if self.check_data_splits_exist():
                self.log_and_print(f"Data splits already exist.")
                if force_write:
                    self.log_and_print(f"Overwriting data splits.")
                else:
                    self.log_and_print(f"Skipping writing data splits.")
                    return
            self.log_and_print(f"Writing data splits to file.")
            if self.data_split_type == "random_ho" or self.data_split_type == "user_ho":
                self.data_splits["train"].to_csv(
                    f"{self.atomic_folder_path}/{self.data_set_name}.train_split.inter", index=False)
                self.data_splits["valid"].to_csv(
                    f"{self.atomic_folder_path}/{self.data_set_name}.valid_split.inter", index=False)
                self.data_splits["test"].to_csv(
                    f"{self.atomic_folder_path}/{self.data_set_name}.test_split.inter", index=False)
            elif self.data_split_type == "random_cv" or self.data_split_type == "user_cv":
                for fold in self.data_splits:
                    self.data_splits[fold]["train"].to_csv(
                        f"{self.atomic_folder_path}/{self.data_set_name}.train_split_{fold}.inter", index=False)
                    self.data_splits[fold]["valid"].to_csv(
                        f"{self.atomic_folder_path}/{self.data_set_name}.valid_split_{fold}.inter", index=False)
                    self.data_splits[fold]["test"].to_csv(
                        f"{self.atomic_folder_path}/{self.data_set_name}.test_split_{fold}.inter", index=False)
            pass

    def load_data_splits(self, force_load=False):
        if self.data_origin == "processed":
            if self.check_data_splits_loaded():
                if force_load:
                    self.log_and_print("Unloading data splits.")
                    self.data_splits = None
                else:
                    self.log_and_print("Skipping data splits loading.")
                    return
            self.log_and_print(f"Loading processed data splits.")
            if self.data_split_type == "random_ho" or self.data_split_type == "user_ho":
                self.data_splits = {
                    "train": pd.read_csv(f"{self.processed_folder_path}/train_split.csv"),
                    "valid": pd.read_csv(f"{self.processed_folder_path}/valid_split.csv"),
                    "test": pd.read_csv(f"{self.processed_folder_path}/test_split.csv")
                }
            elif self.data_split_type == "random_cv" or self.data_split_type == "user_cv":
                self.data_splits = {}
                for file in self.processed_folder_path.iterdir():
                    if "split" in file.name.split("/")[-1]:
                        fold = file.name.split("_")[-1].split(".")[0]
                        self.data_splits[fold] = {
                            "train": pd.read_csv(f"{self.processed_folder_path}/train_split_{fold}.csv"),
                            "valid": pd.read_csv(f"{self.processed_folder_path}/valid_split_{fold}.csv"),
                            "test": pd.read_csv(f"{self.processed_folder_path}/test_split_{fold}.csv")
                        }
            self.log_and_print(f"Loaded data splits.")
        elif self.data_origin == "atomic":
            if self.check_data_splits_loaded():
                if force_load:
                    self.log_and_print("Unloading data splits.")
                    self.data_splits = None
                else:
                    self.log_and_print("Skipping data splits loading.")
                    return
            self.log_and_print(f"Loading atomic data splits.")
            if self.data_split_type == "random_ho" or self.data_split_type == "user_ho":
                self.data_splits = {
                    "train": pd.read_csv(f"{self.atomic_folder_path}/{self.data_set_name}.train_split.inter"),
                    "valid": pd.read_csv(f"{self.atomic_folder_path}/{self.data_set_name}.valid_split.inter"),
                    "test": pd.read_csv(f"{self.atomic_folder_path}/{self.data_set_name}.test_split.inter")
                }
            elif self.data_split_type == "random_cv" or self.data_split_type == "user_cv":
                self.data_splits = {}
                for file in self.atomic_folder_path.iterdir():
                    if "split" in file.name.split("/")[-1]:
                        fold = file.name.split("_")[-1].split(".")[0]
                        self.data_splits[fold] = {
                            "train": pd.read_csv(
                                f"{self.atomic_folder_path}/{self.data_set_name}.train_split_{fold}.inter"),
                            "valid": pd.read_csv(
                                f"{self.atomic_folder_path}/{self.data_set_name}.valid_split_{fold}.inter"),
                            "test": pd.read_csv(
                                f"{self.atomic_folder_path}/{self.data_set_name}.test_split_{fold}.inter")
                        }

    def write_metadata(self, force_write=False):
        if self.meta_data is None:
            self.log_and_print(f"Metadata not calculated.")
            return
        if self.data_origin == "processed":
            if self.processed_data_metadata_exists():
                self.log_and_print(f"Processed metadata already exists.")
                if force_write:
                    self.log_and_print(f"Overwriting processed metadata.")
                else:
                    self.log_and_print(f"Skipping writing processed metadata.")
                    return
            self.log_and_print(f"Writing processed metadata to file.")
            with self.processed_data_metadata_path.open('w') as file:
                json.dump(self.meta_data, file, indent=4)
            self.log_and_print(f"Written processed metadata to file.")
        elif self.data_origin == "atomic":
            if self.atomic_data_metadata_exists():
                self.log_and_print(f"Atomic metadata already exists.")
                if force_write:
                    self.log_and_print(f"Overwriting atomic metadata.")
                else:
                    self.log_and_print(f"Skipping writing atomic metadata.")
                    return
            self.log_and_print(f"Writing atomic metadata to file.")
            with self.atomic_data_metadata_path.open('w') as file:
                json.dump(self.meta_data, file, indent=4)
            self.log_and_print(f"Written atomic metadata to file.")
        else:
            self.log_and_print(f"Wrong data origin. Unable to write metadata.")

    def load_metadata(self, force_load=False):
        if self.data_origin == "processed":
            if self.processed_data_metadata_exists():
                if force_load:
                    self.unload_data()
                else:
                    self.log_and_print(f"Skipping processed metadata loading.")
                    return
                self.log_and_print(f"Loading processed metadata.")
                with self.processed_data_metadata_path.open() as file:
                    self.meta_data = json.load(file)
                self.log_and_print(f"Loaded processed metadata.")
            else:
                self.log_and_print(f"Processed metadata does not exist.")
                return None
        elif self.data_origin == "atomic":
            if self.atomic_data_metadata_exists():
                if force_load:
                    self.unload_data()
                else:
                    self.log_and_print(f"Skipping atomic metadata loading.")
                    return
                self.log_and_print(f"Loading atomic metadata.")
                with self.atomic_data_metadata_path.open() as file:
                    self.meta_data = json.load(file)
                self.log_and_print(f"Loaded atomic metadata.")
            else:
                self.log_and_print(f"Atomic metadata does not exist.")
                return None

    def unload_data(self):
        self.log_and_print(f"Unloading data set.")
        self.data = None
        self.data_origin = None
        self.meta_data = None
        self.data_splits = None
        self.data_split_type = None
        self.feedback_type = None
        self.log_and_print(f"Unloaded data set.")

    def clear_data(self, safety_flag=False):
        if not safety_flag:
            self.log_and_print(f"Set the safety flag to clear data.")
            return
        if self.data_origin == "processed":
            if self.processed_folder_exists():
                self.log_and_print(f"Clearing processed data.")
                for file in self.processed_folder_path.iterdir():
                    self.log_and_print(f"Removing file {file.name}.")
                    file.unlink(missing_ok=True)
                self.processed_folder_path.rmdir()
                self.log_and_print(f"Cleared processed data.")
            else:
                self.log_and_print(f"Processed data does not exist.")
        elif self.data_origin == "atomic":
            if self.atomic_folder_exists():
                self.log_and_print(f"Clearing atomic data.")
                for file in self.atomic_folder_path.iterdir():
                    self.log_and_print(f"Removing file {file.name}.")
                    file.unlink(missing_ok=True)
                self.atomic_folder_path.rmdir()
                self.log_and_print(f"Cleared atomic data.")
            else:
                self.log_and_print(f"Atomic data does not exist.")

    def clear_splits(self):
        if self.data_origin == "processed":
            self.log_and_print(f"Clearing processed data splits.")
            for file in self.processed_folder_path.iterdir():
                if "split" in file.name.split("/")[-1]:
                    file.unlink(missing_ok=True)
            self.log_and_print(f"Cleared processed data splits.")
        elif self.data_origin == "atomic":
            self.log_and_print(f"Clearing atomic data splits.")
            for file in self.atomic_folder_path.iterdir():
                if "split" in file.name.split("/")[-1]:
                    file.unlink(missing_ok=True)
            self.log_and_print(f"Cleared atomic data splits.")
        else:
            self.log_and_print(f"Wrong data origin. Unable to clear splits.")

    # endregion

    # region Logging

    def log_function_time(self, function, *args, **kwargs):
        start_time = time.time_ns()
        function(*args, **kwargs)
        end_time = time.time_ns()
        self.log_and_print(
            f"Function {function.__name__} executed in {(end_time - start_time) // 1_000_000} milliseconds.")

    def log_and_print(self, message):
        self.log.append(message)
        print(message)

    def release_log(self, path):
        with path.open(mode="a") as file:
            for message in self.log:
                file.write(message + "\n")
        self.log = []

    # endregion

    # region Data Set General Processing

    def process_data(self, force_process=False, drop_duplicates=True, normalize_identifiers=True):
        if self.data_origin == "raw":
            if self.processed_data_exists():
                self.log_and_print(f"Processed data set already exists.")
                if force_process:
                    self.log_and_print(f"Continuing raw data to processed data conversion.")
                else:
                    self.log_and_print(f"Skipping raw data to processed data conversion.")
                    return
            self.log_and_print(f"Converting raw data to processed data.")
            Path.mkdir(Path(self.processed_folder_path), exist_ok=True)
            self.load_data()
            if drop_duplicates:
                self.drop_duplicates()
            if normalize_identifiers:
                self.normalize_identifiers()
            self.check_and_order_columns()
            self.check_and_convert_data_types()
            self.log_and_print(f"Processed raw data.")
            self.meta_data = None
            self.data_origin = "processed"
        elif self.data_origin == "processed":
            if self.atomic_data_exists():
                self.log_and_print(f"Atomic data set already exists.")
                if force_process:
                    self.log_and_print(f"Continuing processed data to atomic data conversion.")
                else:
                    self.log_and_print(f"Skipping processed data to atomic data conversion.")
                    return
            self.log_and_print(f"Converting processed data to atomic data.")
            Path.mkdir(Path(self.atomic_folder_path), exist_ok=True)
            self.load_data()
            rename_dict = {self.user_column_name: "user_id:token", self.item_column_name: "item_id:token"}
            if self.rating_column_name in list(self.data):
                rename_dict[self.rating_column_name] = "rating:float"
            if self.timestamp_column_name in list(self.data):
                self.data = self.data.drop(columns=[self.timestamp_column_name]).rename(columns=rename_dict)
            else:
                self.data = self.data.rename(columns=rename_dict)
            self.user_column_name = "user_id:token"
            self.item_column_name = "item_id:token"
            self.rating_column_name = "rating:float"
            self.timestamp_column_name = None
            self.meta_data = None
            self.data_origin = "atomic"

    def set_feedback_type(self):
        if self.rating_column_name in list(self.data):
            self.feedback_type = "explicit"
        else:
            self.feedback_type = "implicit"

    def check_and_order_columns(self):
        self.log_and_print(f"Checking and ordering columns.")
        columns = list(self.data)
        self.log_and_print(f"Data columns: {columns}")
        num_columns = len(columns)
        if self.user_column_name not in columns or self.item_column_name not in columns:
            self.log_and_print(f"Data does not contain {self.user_column_name} and {self.item_column_name} columns.")
            return False

        if num_columns == 2:
            self.log_and_print(f"Data contains {self.user_column_name} and {self.item_column_name} columns.")
            self.data = self.data[[self.user_column_name, self.item_column_name]]
            return True
        elif num_columns == 3:
            if self.rating_column_name in columns:
                self.log_and_print(f"Data contains {self.user_column_name}, {self.item_column_name} and "
                                   f"{self.rating_column_name} columns.")
                self.data = self.data[[self.user_column_name, self.item_column_name, self.rating_column_name]]
                return True
            elif self.timestamp_column_name in columns:
                self.log_and_print(f"Data contains {self.user_column_name}, {self.item_column_name} and "
                                   f"{self.timestamp_column_name} columns.")
                self.data = self.data[[self.user_column_name, self.item_column_name, self.timestamp_column_name]]
                return True
            else:
                self.log_and_print(f"Data contains three columns but none of them is "
                                   f"{self.rating_column_name} or {self.timestamp_column_name}.")
        elif num_columns == 4:
            if self.rating_column_name in columns and self.timestamp_column_name in columns:
                self.log_and_print(f"Data contains {self.user_column_name}, {self.item_column_name}, "
                                   f"{self.rating_column_name} and {self.timestamp_column_name} columns.")
                self.data = self.data[[self.user_column_name, self.item_column_name, self.rating_column_name,
                                       self.timestamp_column_name]]
                return True
            else:
                if self.rating_column_name not in columns:
                    self.log_and_print(f"Data contains four columns but none of them is {self.rating_column_name}.")
                if self.timestamp_column_name not in columns:
                    self.log_and_print(f"Data contains four columns but none of them is {self.timestamp_column_name}.")
                return False
        else:
            self.log_and_print(f"Data contains more than four columns.")
            return False

    def check_and_convert_data_types(self):
        self.log_and_print(f"Checking and converting data types.")
        self.log_and_print(f"Converting data type of column {self.user_column_name} from "
                           f"{self.data[self.user_column_name].dtype} to str.")
        self.data[self.user_column_name] = self.data[self.user_column_name].astype(str)
        self.log_and_print(f"Converting data type of column {self.item_column_name} from "
                           f"{self.data[self.item_column_name].dtype} to str.")
        self.data[self.item_column_name] = self.data[self.item_column_name].astype(str)
        if self.timestamp_column_name in list(self.data):
            self.log_and_print(f"Converting data type of column {self.timestamp_column_name} from "
                               f"{self.data[self.timestamp_column_name].dtype} to str.")
            self.data[self.timestamp_column_name] = self.data[self.timestamp_column_name].astype(str)
        if self.rating_column_name in list(self.data):
            self.log_and_print(f"Converting data type of column {self.rating_column_name} from "
                               f"{self.data[self.rating_column_name].dtype} to np.float64.")
            self.data[self.rating_column_name] = self.data[self.rating_column_name].astype(np.float64)

    # endregion

    # region Data Set Preprocessing

    def drop_duplicates(self):
        self.log_and_print(f"Dropping duplicate interactions.")
        self.log_and_print(f"Number of interactions before: {self.num_interactions()}")
        self.data.drop_duplicates(subset=[self.user_column_name, self.item_column_name], keep="last", inplace=True)
        self.log_and_print(f"Number of interactions after: {self.num_interactions()}")

    def make_implicit(self, threshold: int | float):
        if self.feedback_type == "implicit":
            self.log_and_print(f"Data set already implicit.")
            return
        self.log_and_print(f"Making data set implicit with threshold {threshold}.")
        self.log_and_print(f"Minimum rating: {self.min_rating()}")
        self.log_and_print(f"Maximum rating: {self.max_rating()}")
        self.log_and_print(f"Number of interactions before: {self.num_interactions()}")
        if isinstance(threshold, int):
            self.data = self.data[self.data[self.rating_column_name] >= threshold][
                [self.user_column_name, self.item_column_name]]
        elif isinstance(threshold, float) and (0 <= threshold <= 1):
            scaled_max_rating = abs(self.max_rating()) + abs(self.min_rating())
            rating_cutoff = round(scaled_max_rating * threshold) - abs(self.min_rating())
            self.data = self.data[self.data[self.rating_column_name] >= rating_cutoff][
                [self.user_column_name, self.item_column_name]]
        else:
            self.log_and_print(f"Threshold must be an integer or a float between 0 and 1.")
        self.set_feedback_type()
        self.log_and_print(f"Number of interactions after: {self.num_interactions()}")

    def core_pruning(self, core: int):
        self.log_and_print(f"Pruning data set to {core}-core.")
        self.log_and_print(f"Number of interactions before: {self.num_interactions()}")

        def parallel_count():
            shared_dict = Manager().dict()
            u_cnt_job = Process(target=count_elements,
                                args=(self.data[self.user_column_name].values, self.user_column_name, shared_dict))
            u_cnt_job.start()
            i_cnt_job = Process(target=count_elements,
                                args=(self.data[self.item_column_name].values, self.item_column_name, shared_dict))
            i_cnt_job.start()

            u_cnt_job.join()
            i_cnt_job.join()

            u_cnt, i_cnt = shared_dict.values()[0], shared_dict.values()[1]

            return u_cnt, i_cnt

        uc, ic = parallel_count()

        while len(self.data) > 0 and min(uc[1]) < core or min(ic[1]) < core:
            shared_dict_outer = Manager().dict()
            u_sig_job = Process(target=filter_invalid, args=(
            self.data[self.user_column_name].values, uc, self.user_column_name, core, shared_dict_outer))
            u_sig_job.start()
            i_sig_job = Process(target=filter_invalid, args=(
            self.data[self.item_column_name].values, ic, self.item_column_name, core, shared_dict_outer))
            i_sig_job.start()
            u_sig_job.join()
            i_sig_job.join()

            user_valid, item_valid = shared_dict_outer.values()[0], shared_dict_outer.values()[1]
            self.data = self.data[(user_valid & item_valid)]

            uc, ic = parallel_count()

        # u_sig = [k for k in uc if (uc[k] < len(self.data[self.item_column_name].unique()))]
        # self.data = self.data[self.data[self.user_column_name].isin(u_sig)]

        self.log_and_print(f"Number of interactions after: {self.num_interactions()}")

    def subsample(self, sample_size: int, random_state: int = 42):
        if len(self.data) <= sample_size:
            self.log_and_print(f"Data set has less interactions than the sample size. Unable to subsample.")
            return
        self.log_and_print(f"Subsampling data set to {sample_size} interactions.")
        self.log_and_print(f"Number of interactions before: {self.num_interactions()}")
        self.data = self.data.sample(n=sample_size, random_state=random_state)
        self.log_and_print(f"Number of interactions after: {self.num_interactions()}")

    def normalize_identifiers(self):
        self.log_and_print(f"Normalizing identifiers.")
        for col in [self.user_column_name, self.item_column_name]:
            unique_ids = {key: value for value, key in enumerate(self.data[col].unique())}
            self.data[col].update(self.data[col].map(unique_ids))

    def split_data(self, num_folds=None, valid_size=None, test_size=None, random_state=None):
        if self.data_split_type == "random_ho":
            train, test = train_test_split(self.data, test_size=test_size, random_state=random_state)
            train, valid = train_test_split(train, test_size=valid_size / (1 - test_size), random_state=random_state)
            self.data_splits = {"train": train, "valid": valid, "test": test}
        if self.data_split_type == "random_cv":
            self.data_splits = {}
            folds = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            for fold, (train_index, test_index) in enumerate(folds.split(self.data)):
                train, test = self.data.iloc[train_index], self.data.iloc[test_index]
                train, valid = train_test_split(train, test_size=valid_size / (1 - (1 / num_folds)),
                                                random_state=random_state)
                self.data_splits[f"fold_{fold}"] = {"train": train, "valid": valid, "test": test}
        if self.data_split_type == "user_ho":
            indices = {"train": np.array([]), "valid": np.array([]), "test": np.array([])}
            self.data.reset_index(drop=True, inplace=True)
            for user, items in self.data.groupby(self.user_column_name).indices.items():
                train, test = train_test_split(items, test_size=test_size, random_state=random_state)
                train, valid = train_test_split(train, test_size=valid_size / (1 - test_size),
                                                random_state=random_state)
                indices["train"] = np.append(indices["train"], train)
                indices["valid"] = np.append(indices["valid"], valid)
                indices["test"] = np.append(indices["test"], test)
            self.data_splits = {"train": self.data.iloc[indices["train"]], "valid": self.data.iloc[indices["valid"]],
                                "test": self.data.iloc[indices["test"]]}
            if self.data_split_type == "user_cv":
                fold_indices = {i: {"train": np.array([]), "valid": np.array([]), "test": np.array([])} for i in
                                range(num_folds)}
                rng = np.random.default_rng(random_state)
                self.data.reset_index(drop=True, inplace=True)
                for user, items in self.data.groupby(self.user_column_name).indices.items():
                    if len(items) < num_folds:
                        self.log_and_print(
                            f"User {user} has less interactions than the number of folds. Unable to split.")
                        return
                    rng.shuffle(items)
                    splits = np.array_split(items, num_folds)
                    for i in range(len(splits)):
                        train_full = np.concatenate(splits[:i] + splits[i + 1:])
                        train, valid = train_test_split(train_full, test_size=valid_size / (1 - (1 / num_folds)),
                                                        random_state=random_state)
                        if len(valid) < 1:
                            self.log_and_print("Validation set is empty. Unable to split.")
                            return
                        test = splits[i]
                        fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train)
                        fold_indices[i]["valid"] = np.append(fold_indices[i]["valid"], valid)
                        fold_indices[i]["test"] = np.append(fold_indices[i]["test"], test)
                self.data_splits = {}
                for fold in range(num_folds):
                    self.data_splits[f"fold_{fold}"] = {"train": self.data.iloc[fold_indices[fold]["train"]],
                                                        "valid": self.data.iloc[fold_indices[fold]["valid"]],
                                                        "test": self.data.iloc[fold_indices[fold]["test"]]}

    # endregion

    # region Data Set Statistics

    def min_rating(self):
        return self.data[self.rating_column_name].min() if self.feedback_type == "explicit" else None

    def max_rating(self):
        return self.data[self.rating_column_name].max() if self.feedback_type == "explicit" else None

    def mean_rating(self):
        return self.data[self.rating_column_name].mean() if self.feedback_type == "explicit" else None

    def normalized_mean_rating(self):
        return (self.mean_rating() - self.min_rating()) / (self.max_rating() - self.min_rating()) \
            if self.feedback_type == "explicit" else None

    def median_rating(self):
        return self.data[self.rating_column_name].median() if self.feedback_type == "explicit" else None

    def normalized_median_rating(self):
        return (self.median_rating() - self.min_rating()) / (self.max_rating() - self.min_rating()) \
            if self.feedback_type == "explicit" else None

    def mode_rating(self):
        return self.data[self.rating_column_name].mode().iloc[0] if self.feedback_type == "explicit" else None

    def normalized_mode_rating(self):
        return (self.mode_rating() - self.min_rating()) / (self.max_rating() - self.min_rating()) \
            if self.feedback_type == "explicit" else None

    def num_users(self):
        return len(self.data[self.user_column_name].unique())

    def user_counter(self):
        return Counter(self.data[self.user_column_name])

    def highest_num_rating_by_single_user(self):
        return self.user_counter().most_common()[0][1]

    def lowest_num_rating_by_single_user(self):
        return self.user_counter().most_common()[-1][1]

    def mean_num_ratings_by_user(self):
        return self.num_interactions() / self.num_users()

    def num_items(self):
        return len(self.data[self.item_column_name].unique())

    def item_counter(self):
        return Counter(self.data[self.item_column_name])

    def highest_num_rating_on_single_item(self):
        return self.item_counter().most_common()[0][1]

    def lowest_num_rating_on_single_item(self):
        return self.item_counter().most_common()[-1][1]

    def mean_num_ratings_on_item(self):
        return self.num_interactions() / self.num_items()

    def user_item_ratio(self):
        return self.num_users() / self.num_items()

    def item_user_ratio(self):
        return self.num_items() / self.num_users()

    def density(self):
        return self.num_interactions() / (self.num_users() * self.num_items()) * 100

    def num_interactions(self):
        return len(self.data)

    def num_possible_ratings(self):
        return len(self.data[self.rating_column_name].unique())

    def calculate_metadata(self, force_calculate=False):
        if not self.check_data_loaded():
            return
        if self.data_origin == "processed" or self.data_origin == "atomic":
            if self.data_origin == "processed":
                if self.processed_data_metadata_exists():
                    if force_calculate:
                        self.log_and_print(f"Overwriting processed metadata.")
                    else:
                        self.log_and_print(f"Skipping processed metadata calculation.")
                        return
            elif self.data_origin == "atomic":
                if self.atomic_data_metadata_exists():
                    if force_calculate:
                        self.log_and_print(f"Overwriting atomic metadata.")
                    else:
                        self.log_and_print(f"Skipping atomic metadata calculation.")
                        return
            self.log_and_print(f"Calculating metadata.")
            '''          
            "min_rating": float(self.min_rating()),
            "max_rating": float(self.max_rating()),
            "mean_rating": float(self.mean_rating()),
            "median_rating": float(self.median_rating()),
            "mode_rating": float(self.mode_rating()),
            "normalized_mean_rating": float(self.normalized_mean_rating()),
            "normalized_median_rating": float(self.normalized_median_rating()),
            "normalized_mode_rating": float(self.normalized_mode_rating()),           
            "num_possible_ratings": self.num_possible_ratings(),
            '''
            self.meta_data = {
                "num_users": self.num_users(),
                "num_items": self.num_items(),
                "num_interactions": self.num_interactions(),
                "density": self.density(),
                "feedback_type": self.feedback_type,
                "user_item_ratio": self.user_item_ratio(),
                "item_user_ratio": self.item_user_ratio(),
                "highest_num_rating_by_single_user": self.highest_num_rating_by_single_user(),
                "lowest_num_rating_by_single_user": self.lowest_num_rating_by_single_user(),
                "highest_num_rating_on_single_item": self.highest_num_rating_on_single_item(),
                "lowest_num_rating_on_single_item": self.lowest_num_rating_on_single_item(),
                "mean_num_ratings_by_user": self.mean_num_ratings_by_user(),
                "mean_num_ratings_on_item": self.mean_num_ratings_on_item()
            }
            self.log_and_print(f"Calculated metadata.")
        else:
            self.log_and_print(f"Data origin is not processed or atomic. Unable to calculate metadata.")

        # endregion
