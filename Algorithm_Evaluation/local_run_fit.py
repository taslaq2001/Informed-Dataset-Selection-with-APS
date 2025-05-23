
import os
#from hpc_data_set_names import data_set_names
from hpc_algorithm_names import algorithm_names
data_set_names =['airlines','animes','bank','beer','booksthird','clothes','coffee','homeworkouts','hotels','hotelssecond','iphone','laptops','mcdonalds','products','productssecond','ramen','sales','sephora','starbucks','uber','videogames','videogamessecond']
folds = [0, 1, 2, 3, 4]

for data_set_name in data_set_names:
    for algorithm in algorithm_names:
        for fold in folds:
            cmd = f"python run_recbole_fit.py --data_set_name {data_set_name} --algorithm_name {algorithm} --algorithm_config 0 --fold {fold}"
            print(f"Running: {cmd}")
            os.system(cmd)
