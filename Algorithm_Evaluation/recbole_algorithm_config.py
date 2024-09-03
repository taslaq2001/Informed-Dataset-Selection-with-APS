import itertools


def retrieve_configurations(algorithm_name):
    configuration_space = {}
    if algorithm_name == "Pop":
        configuration_space["epochs"] = [1]
    elif algorithm_name == "ItemKNN":
        configuration_space["epochs"] = [1]
        configuration_space["k"] = [100]
    elif algorithm_name == "BPR":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "NeuMF":
        configuration_space["mf_embedding_size"] = [64]
        configuration_space["mlp_embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "ConvNCF":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "DMF":
        configuration_space["user_embedding_size"] = [64]
        configuration_space["item_embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "FISM":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
        configuration_space["split_to"] = [5]
    elif algorithm_name == "NAIS":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
        configuration_space["split_to"] = [20]
    elif algorithm_name == "SpectralCF":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "GCMC":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "NGCF":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "LightGCN":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "DGCF":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "LINE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "MultiVAE":
        configuration_space["latent_dimension"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "MultiDAE":
        configuration_space["latent_dimension"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "MacridVAE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "CDAE":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "ENMF":
        configuration_space["embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "NNCF":
        configuration_space["ui_embedding_size"] = [64]
        configuration_space["neigh_embedding_size"] = [64]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "RecVAE":
        configuration_space["latent_dimension"] = [200]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "EASE":
        configuration_space["epochs"] = [1]
        configuration_space["reg_weight"] = [250]
    elif algorithm_name == "SGL":
        configuration_space["embedding_size"] = [64]
    elif algorithm_name == "NCEPLRec":
        configuration_space["epochs"] = [1]
        configuration_space["rank"] = [450]
    elif algorithm_name == "SimpleX":
        configuration_space["embedding_size"] = [64]
    elif algorithm_name == "NCL":
        configuration_space["embedding_size"] = [64]
        configuration_space["num_clusters"] = [100]
    elif algorithm_name == "Random":
        configuration_space["epochs"] = [1]
    elif algorithm_name == "DiffRec":
        configuration_space["steps"] = [5]
        configuration_space["learning_rate"] = [0.01]
    elif algorithm_name == "LDiffRec":
        configuration_space["steps"] = [5]
        configuration_space["learning_rate"] = [0.01]

    experiments = [dict(zip(configuration_space.keys(), v)) for v in itertools.product(*configuration_space.values())]
    return experiments
