from hpc_data_set_names import data_set_names
import os

for name in data_set_names:
    print(f"\nProcessing dataset: {name}")
    os.system(f"python run_convert_raw_to_processed.py --data_set_name {name}")
