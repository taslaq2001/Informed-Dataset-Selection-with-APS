import itertools

def retrieve_configurations(algorithm_name):
    configuration_space = {}

    if algorithm_name == "NFM":
        configuration_space["n_factors"] = [15, 30]
        configuration_space["n_epochs"] = [10]
    elif algorithm_name == "ADMM-Slim":
        configuration_space["n_factors"] = [10, 20, 50]  # Latent factors
        configuration_space["n_epochs"] = [10, 20]  # Number of training epochs
        configuration_space["lambda_reg"] = [0.1, 0.01, 0.001]  # Regularization term for sparsity
        configuration_space["learning_rate"] = [0.01, 0.001]  # Learning rate
        configuration_space["batch_size"] = [64, 128, 256]  # Mini-batch size (if using mini-batch ADMM)
        configuration_space["alpha"] = [1.0, 0.5]  # ADMM specific parameter (regularization weight)
        configuration_space["rho"] = [1.0, 2.0]  # ADMM specific parameter (stepsize)    
    elif algorithm_name == "SlopeOne":
        configuration_space["dummy"] = [0]  
    elif algorithm_name == "CoClustering":
        configuration_space["n_cltr_u"] = [3]
        configuration_space["n_cltr_i"] = [3]
        configuration_space["n_epochs"] = [10]
    elif algorithm_name == "SVDpp":
        configuration_space["n_factors"] = [20]
        configuration_space["n_epochs"] = [10]
    elif algorithm_name == "TFIDF_Cosine":
        configuration_space["top_k"] = [20, 50]
    elif algorithm_name == "ALS":
        configuration_space["factors"] = [50]
        configuration_space["regularization"] = [0.01]
        configuration_space["iterations"] = [10]
    elif algorithm_name == "Autoencoder":
        configuration_space["hidden_dim"] = [64, 128]
        configuration_space["learning_rate"] = [0.001]
        configuration_space["epochs"] = [10]
    elif algorithm_name == "FM":
        configuration_space["task"] = ["rating"]
        configuration_space["epoch"] = [10]
        configuration_space["lr"] = [0.05]
    elif algorithm_name == "Apriori":
        configuration_space["min_support"] = [0.01]
    elif algorithm_name == "UserBasedCF":
        configuration_space["k"] = [20]
        configuration_space["sim_option"] = [{"name": "cosine", "user_based": True}]

    experiments = [dict(zip(configuration_space.keys(), v)) for v in itertools.product(*configuration_space.values())]
    return experiments
