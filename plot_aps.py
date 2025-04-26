
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import math

if __name__ == "__main__":
    metric = "NDCG@10"
    highlight = True

    data = pd.read_csv("results.csv", usecols=["data_set_name", "algorithm_name", metric, "num_interactions"])
    alg_list = data["algorithm_name"].unique().tolist()
    data = data.groupby(["data_set_name", "algorithm_name"])[metric].mean().to_frame().reset_index()
    data = data[data["data_set_name"] != "Amazon2018-Fashion"]
    # data.to_csv("aps.csv", index=False)
    # data.to_excel("aps.xlsx", index=False)

    with PdfPages("APS.pdf") as pdf:
        for alg1 in alg_list:
            h = 0
            w = 0
            dfx = data[data["algorithm_name"] == alg1]
            height = math.ceil((len(alg_list) - 1) / 7)
            width = math.ceil((len(alg_list) - 1) / height)
            fig, axs = plt.subplots(height, width, layout="constrained", figsize=(16, 9))
            for alg2 in alg_list:
                if alg1 == alg2:
                    continue
                dfy = data[data["algorithm_name"] == alg2]
                merged = pd.merge(dfx, dfy, on="data_set_name")
                x = merged[f"{metric}_x"]
                y = merged[f"{metric}_y"]
                max_total = max(x.max(), y.max())

                # seperate movielens and amazon data for highlighting
                if highlight:
                    movielens = merged.query("data_set_name.str.contains('movielens', case=False)")
                    amazon = merged.query("data_set_name.str.contains('amazon', case=False)")
                    merged = pd.merge(merged, amazon, indicator=True, how="outer").query("_merge=='left_only'").drop("_merge", axis=1)
                    merged = pd.merge(merged, movielens, indicator=True, how="outer").query("_merge=='left_only'").drop("_merge", axis=1)

                    # normalize
                    xm = movielens[f"{metric}_x"] / max_total
                    ym = movielens[f"{metric}_y"]  / max_total
                    xa = amazon[f"{metric}_x"] / max_total
                    ya = amazon[f"{metric}_y"] / max_total

                x = merged[f"{metric}_x"] / max_total
                y = merged[f"{metric}_y"] / max_total
                axs[h, w].set(xlabel=alg1, ylabel=alg2)
                axs[h, w].set_xlim(0, 1)
                axs[h, w].set_ylim(0, 1)
                axs[h, w].set_aspect("equal", "box")
                axs[h, w].plot([0, 1], [0, 1])
                axs[h, w].plot(x, y, "o", markersize=4)

                if highlight:
                    axs[h, w].plot(xm, ym, "mx", markersize=7)
                    axs[h, w].plot(xa, ya, "k2", markersize=7)
                w = (w + 1) % 7
                if w == 0:
                    h += 1

            # plt.show()
            plt.title(alg1)
            pdf.savefig(fig)
            plt.close()
            print(f"Page done for {alg1}.")
        d = pdf.infodict()
        d['Title'] = 'Algorithm Performance Spaces'

