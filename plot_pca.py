

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.colors as mcolors

# List of datasets
data_set_names = [
    'adidasvsnike','airlines','airlinesreviews','animes','books','clothes',
    'disneylandreviews','hotelssecond','iphone','laptops','products',
    'productssecond','rottentomatoesmovies','ryanairreviews','starbucks',
    'videogamessecond','winesspa'
]

# Use distinct named colors from CSS4_COLORS
available_colors = list(mcolors.CSS4_COLORS.keys())
excluded_colors = {"white", "whitesmoke", "snow", "azure", "ivory", "aliceblue"}
filtered_colors = [c for c in available_colors if c.lower() not in excluded_colors]
assert len(filtered_colors) >= len(data_set_names), "Not enough distinct colors for datasets."

# Assign unique readable colors
dataset_colors = {ds: filtered_colors[i] for i, ds in enumerate(data_set_names)}

# Print assigned color names
print("Dataset color names:")
for ds, color in dataset_colors.items():
    print(f"{ds}: {color}")

if __name__ == "__main__":
    metric = "NDCG@10"
    label = True
    label_cutoff_x = 0.2
    save_fig = True

    # Load and preprocess
    data = pd.read_csv("merged.csv", usecols=["data_set_name", "algorithm_name", metric])
    data = data.groupby(["data_set_name", "algorithm_name"])[metric].mean().to_frame().reset_index()
    data = data[data["data_set_name"].isin(data_set_names)]
    data = data[data["algorithm_name"] != "Random"]

    algorithms = data["algorithm_name"].unique().tolist()
    datasets = data["data_set_name"].unique().tolist()

    # Build data matrix
    X = []
    for ds in datasets:
        row = []
        for alg in algorithms:
            result = data.loc[(data["algorithm_name"] == alg) & (data["data_set_name"] == ds), metric]
            row.append(result.iat[0] if not result.empty else np.nan)
        X.append(row)

    # Impute missing values and apply PCA
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    X = imp.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    variance = pca.explained_variance_ratio_

    # Plot
    plt.figure(figsize=(10, 7))
    for xy, ds in zip(X_pca, datasets):
        color = dataset_colors.get(ds, "gray")
        plt.plot(xy[0], xy[1], "o", markersize=5, color=color)
        if label and xy[0] > label_cutoff_x:
            plt.text(xy[0], xy[1], ds, fontsize=7)

    plt.xlabel(f"Component 1 - {variance[0]:.2%}")
    plt.ylabel(f"Component 2 - {variance[1]:.2%}")
    plt.gca().set_aspect("auto", "box")

    if save_fig:
        plt.savefig("PCA")

    print(f"Datasets: {len(datasets)}, Algorithms: {len(algorithms)}, "
          f"Variance C1: {variance[0]:.2%}, Variance C2: {variance[1]:.2%}")

    plt.show()
