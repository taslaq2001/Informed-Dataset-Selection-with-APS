import argparse
import json
import time

import numpy as np
import pandas as pd
from recbole.quick_start import load_data_and_model

import torch
from recbole.utils.case_study import full_sort_topk
from recbole_algorithm_config import retrieve_configurations

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Predict RecBole")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--algorithm_name', dest='algorithm_name', type=str, required=True)
    parser.add_argument('--algorithm_config', dest='algorithm_config', type=int, required=True)
    parser.add_argument('--fold', dest='fold', type=int, required=True)

    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"PyTorch version: {torch.__version__}")

    configurations = retrieve_configurations(algorithm_name=args.algorithm_name)

    checkpoint_path = (f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
                       f"config_{args.algorithm_config}/fold_{args.fold}/")
    fit_log_file = (f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
                    f"config_{args.algorithm_config}/fold_{args.fold}/fit_log.json")
    with open(fit_log_file, "r") as file:
        fit_log = json.load(file)
    model_file = fit_log["model_file"]

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)

    train = pd.read_csv(
        f"./data_sets/{args.data_set_name}/atomic/{args.data_set_name}.train_split_fold_{args.fold}.inter",
        header=0, sep=",")
    test = pd.read_csv(
        f"./data_sets/{args.data_set_name}/atomic/{args.data_set_name}.test_split_fold_{args.fold}.inter",
        header=0, sep=",")

    unique_train_users = train["user_id:token"].unique()
    unique_test_users = test["user_id:token"].unique()
    users_to_predict = np.intersect1d(unique_test_users, unique_train_users)
    uid_series = dataset.token2id(dataset.uid_field, list(map(str, users_to_predict)))
    top_k_score = []
    top_k_iid_list = []
    start_prediction = time.time()
    for uid in uid_series:
        uid_top_k_score, uid_top_k_iid_list = full_sort_topk(np.array([uid]), model, test_data, k=20,
                                                             device=config['device'])
        # convert tensor to numpy array and then to list
        top_k_score.append(uid_top_k_score.cpu().numpy().tolist()[0])
        top_k_iid_list.append(uid_top_k_iid_list.cpu().numpy().tolist()[0])
    end_prediction = time.time()
    # make dictionary with uid_series as key and top_k as value
    top_k_dict = dict(zip(uid_series.tolist(), zip(top_k_iid_list, top_k_score)))

    with open(f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
              f"config_{args.algorithm_config}/fold_{args.fold}/predictions.json", "w") as file:
        json.dump(top_k_dict, file, indent=4)

    predict_log_dict = {
        "model_file": model_file,
        "data_set_name": args.data_set_name,
        "algorithm_name": args.algorithm_name,
        "algorithm_config_index": args.algorithm_config,
        "algorithm_configuration": configurations[args.algorithm_config],
        "fold": args.fold,
        "train_users": len(unique_train_users),
        "test_users": len(unique_test_users),
        "users_to_predict": len(users_to_predict),
        "prediction_time": end_prediction - start_prediction
    }

    with open(f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
              f"config_{args.algorithm_config}/fold_{args.fold}/predict_log.json", "w") as file:
        json.dump(predict_log_dict, file, indent=4)