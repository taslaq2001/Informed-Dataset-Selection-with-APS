import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import normalize, MinMaxScaler


if __name__ == "__main__":
    metric = "NDCG@10"
    label = True
    label_cutoff_x = 0.2
    save_fig = True
    highlight = True

    data = pd.read_csv("results.csv", usecols=["data_set_name", "algorithm_name", metric])
    data = data.groupby(["data_set_name", "algorithm_name"])[metric].mean().to_frame().reset_index()
    test = data[data["algorithm_name"] == "BPR"].sort_values(by=[metric])
    data = data[data["data_set_name"] != "Amazon2018-Fashion"]
    data = data[data["algorithm_name"] != "Random"]
        
    algorithms = data["algorithm_name"].unique().tolist()
    datasets = data["data_set_name"].unique().tolist()
    X = []
    for ds in datasets:
        row = []
        for alg in algorithms:
            result = data.loc[(data["algorithm_name"] == alg) & (data["data_set_name"] == ds), metric]
            if len(result) > 0:
                result = result.iat[0]
            else:
                result = np.nan
            row.append(result)
        X.append(row)
    imp = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp.fit(X)
    X = imp.transform(X)
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    variance = pca.explained_variance_ratio_

    df = pd.DataFrame(X_pca, columns=["component_1", "component_2"], index=datasets)
    #df.to_csv("pca.csv")
    #df.to_excel("pca.xlsx")

    plt.gca().set_aspect("auto", "box")
    if highlight:
        movielens = df[df.index.str.contains("movielens", case=False)]
        amazon = df[df.index.str.contains("amazon", case=False)]
        merged = pd.merge(df, amazon, indicator=True, how="outer").query("_merge=='left_only'").drop("_merge", axis=1)
        merged = pd.merge(merged, movielens, indicator=True, how="outer").query("_merge=='left_only'").drop("_merge", axis=1)
        plt.plot("component_1", "component_2", "o", data=merged, markersize=4)
        plt.plot("component_1", "component_2", "mx", data=movielens, markersize=7)
        plt.plot("component_1", "component_2", "k2", data=amazon, markersize=7)
    else:
        plt.plot("component_1", "component_2", "o", data=df, markersize=4)
    if label:
        for xy, ds in zip(X_pca, datasets):
            if xy[0] > label_cutoff_x:
                plt.text(xy[0], xy[1], ds, fontsize=6)
    plt.xlabel(f"Component 1 - {variance[0]:.2%}")
    plt.ylabel(f"Component 2 - {variance[1]:.2%}")
    if save_fig:
        plt.savefig("PCA")
    print(f"Datasets: {len(datasets)}, Algorithms: {len(algorithms)}, Variance C1: {variance[0]:.2%}, Variance C2: {variance[1]:.2%}")
    plt.show()
