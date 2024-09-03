import argparse
import json
import time
from logging import getLogger
import signal
import os

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import ModelType, get_model, get_trainer, init_seed, init_logger

import torch
from recbole_algorithm_config import retrieve_configurations

if __name__ == "__main__":
    setup_start_time = time.time()

    parser = argparse.ArgumentParser("Fit RecBole")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--algorithm_name', dest='algorithm_name', type=str, required=True)
    parser.add_argument('--algorithm_config', dest='algorithm_config', type=int, required=True)
    parser.add_argument('--fold', dest='fold', type=int, required=True)

    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"PyTorch version: {torch.__version__}")

    config_dict = {
        "seed": 42,  # default: "2020"
        "data_path": "./data_sets/",  # default: "dataset/"
        "checkpoint_dir": f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
                          f"config_{args.algorithm_config}/fold_{args.fold}/",  # default: "saved/"
        "log_wandb": False,  # default: False
        "wandb_project": f"{args.data_set_name} @ {args.algorithm_name}",  # default: "recbole"
        "benchmark_filename": [f"train_split_fold_{args.fold}", f"valid_split_fold_{args.fold}",
                               f"test_split_fold_{args.fold}"],
        # default: None
        "field_separator": ",",  # default: "\t"
        "epochs": 50,  # default: 300
        "train_batch_size": 2048,  # default: 2048
        "learner": "adam",  # default: "adam"
        "learning_rate": 0.01,  # default: 0.001
        "training_neg_sample_args":
            {
                "distribution": "uniform",  # default: "uniform"
                "sample_num": 1,  # default: 1
                "dynamic": False,  # default: False
                "candidate_num": 0,  # default: 0
            },
        "eval_step": 100,  # default: 1
        "stopping_step": 100,  # default: 10
        "weight_decay": 0.0,  # default: 0.0
        "eval_args":
            {
                "group_by": "user",  # default: "user"
                "order": "RO",  # default: "RO"
                "split":
                    {
                        # "RS": [8, 1, 1] # default: {"RS": [8, 1, 1]}
                        "LS": "valid_and_test"
                    },
                "mode":
                    {
                        "valid": "full",  # default: "full"
                        "test": "full",  # default: "full"
                    },
            },
        "metrics": ["NDCG"],
        #"metrics": ["Recall", "MRR", "NDCG", "Hit", "MAP", "Precision", "GAUC", "ItemCoverage", "AveragePopularity",
        #            "GiniIndex", "ShannonEntropy", "TailPercentage"],
        # default: ["Recall", "MRR", "NDCG", "Hit", "Precision"]
        "topk": [10],  # default: 10
        "valid_metric": "NDCG@10",  # default: "MRR@10"
        "eval_batch_size": 32768,  # default: 4096
        # misc settings
        "model": args.algorithm_name,
        "MODEL_TYPE": ModelType.GENERAL,  # default: ModelType.GENERAL
        "dataset": args.data_set_name,  # default: None
    }

    configurations = retrieve_configurations(algorithm_name=args.algorithm_name)
    config_dict.update(configurations[args.algorithm_config])

    config = Config(config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    logger.info(f"Running algorithm {args.algorithm_name} configuration: {configurations[args.algorithm_config]}.")

    config["data_path"] = f"./data_sets/{args.data_set_name}/atomic/"
    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    logger.info("Loading model.")
    model = get_model(config["model"])(config, train_data.dataset).to(config['device'])
    logger.info(model)
    logger.info("Loading trainer.")
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    setup_end_time = time.time()
    logger.info(f"Setup time: {setup_end_time - setup_start_time} seconds.")
    remaining_time = 1800 - int((setup_end_time - setup_start_time))
    logger.info(f"Remaining time: {remaining_time} seconds.")
    if remaining_time < 0:
        logger.info(f"Setup exceeded time limit.")

    fit_start_time = time.time()
    if os.name == 'nt':
        trainer.fit(train_data)
        # best_valid_score, best_valid_result = trainer.fit(train_data)
    elif os.name == 'posix':
        def timeout_fit(signum, frame):
            raise TimeoutError("Training exceeded time limit.")
        signal.signal(signal.SIGALRM, timeout_fit)
        signal.alarm(remaining_time)
        try:
            trainer.fit(train_data)
        except TimeoutError:
            logger.info("Training exceeded time limit.")
            pass

    fit_end_time = time.time()
    logger.info(f"Training time: {fit_end_time - fit_start_time} seconds")
    model_file = trainer.saved_model_file

    fit_log_dict = {
        "model_file": model_file,
        "data_set_name": args.data_set_name,
        "algorithm_name": args.algorithm_name,
        "algorithm_config_index": args.algorithm_config,
        "algorithm_configuration": configurations[args.algorithm_config],
        "fold": args.fold,
        #"best_validation_score": best_valid_score,
        "setup_time": setup_end_time - setup_start_time,
        "training_time": fit_end_time - fit_start_time
    }

    with open(f"./data_sets/{args.data_set_name}/checkpoint_{args.algorithm_name}/"
              f"config_{args.algorithm_config}/fold_{args.fold}/fit_log.json", mode="w") as file:
        json.dump(fit_log_dict, file, indent=4)
