import argparse
import json
import time

import pandas as pd
from recbole.quick_start import load_data_and_model

import torch
from recbole_algorithm_config import retrieve_configurations

from run_utils import ndcg, hr, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate RecBole")
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
    predict_log_file = (f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
                        f"config_{args.algorithm_config}/fold_{args.fold}/predict_log.json")
    with open(predict_log_file, "r") as file:
        predict_log = json.load(file)
    model_file = predict_log["model_file"]

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=model_file)

    test = pd.read_csv(
        f"./data_sets/{args.data_set_name}/atomic/{args.data_set_name}.test_split_fold_{args.fold}.inter",
        header=0, sep=",")
    test["user_id:token"] = dataset.token2id(dataset.uid_field, list(map(str, test["user_id:token"].values)))
    test["item_id:token"] = dataset.token2id(dataset.iid_field, list(map(str, test["item_id:token"].values)))

    with open(f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
              f"config_{args.algorithm_config}/fold_{args.fold}/predictions.json", "r") as file:
        top_k_dict = json.load(file)

    top_k_dict = {int(k): v[0] for k, v in top_k_dict.items()}
    k_options = [1, 3, 5, 10, 20]

    start_evaluation = time.time()
    ndcg_per_user_per_k = ndcg(top_k_dict, k_options, test, "user_id:token", "item_id:token")
    hr_per_user_per_k = hr(top_k_dict, k_options, test, "user_id:token", "item_id:token")
    recall_per_user_per_k = recall(top_k_dict, k_options, test, "user_id:token", "item_id:token")
    end_evaluation = time.time()

    evaluate_log_dict = {
        "model_file": model_file,
        "data_set_name": args.data_set_name,
        "algorithm_name": args.algorithm_name,
        "algorithm_config_index": args.algorithm_config,
        "algorithm_configuration": configurations[args.algorithm_config],
        "fold": args.fold,
        "evaluation_time": end_evaluation - start_evaluation
    }

    for k in k_options:
        score = sum(ndcg_per_user_per_k[k]) / len(ndcg_per_user_per_k[k])
        print(f"NDCG@{k}: {score}")
        evaluate_log_dict[f"NDCG@{k}"] = score
        score = sum(hr_per_user_per_k[k]) / len(hr_per_user_per_k[k])
        print(f"HR@{k}: {score}")
        evaluate_log_dict[f"HR@{k}"] = score
        score = sum(recall_per_user_per_k[k]) / len(recall_per_user_per_k[k])
        print(f"Recall@{k}: {score}")
        evaluate_log_dict[f"Recall@{k}"] = score

    with open(f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
              f"config_{args.algorithm_config}/fold_{args.fold}/evaluate_log.json", 'w') as file:
        json.dump(evaluate_log_dict, file, indent=4)
