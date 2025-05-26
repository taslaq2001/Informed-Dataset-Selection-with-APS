
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import math

# List of dataset names
data_set_names = [
    'adidasvsnike', 'airlines', 'airlinesreviews', 'animes', 'books', 'clothes',
    'disneylandreviews', 'hotelssecond', 'iphone', 'laptops', 'products',
    'productssecond', 'rottentomatoesmovies', 'ryanairreviews', 'starbucks',
    'videogamessecond', 'winesspa'
]

# Generate a list of unique colors
all_colors = list(mcolors.CSS4_COLORS.keys())
all_colors = [c for c in all_colors if c.lower() not in ["white", "whitesmoke", "snow", "azure", "ivory"]]  # remove unreadable colors

# Ensure we have enough colors
assert len(all_colors) >= len(data_set_names), "Not enough distinct colors for all datasets."

# Assign unique named colors to each dataset
dataset_colors = {ds: all_colors[i] for i, ds in enumerate(data_set_names)}

# Print assigned color names
print("Dataset color names:")
for ds, color in dataset_colors.items():
    print(f"{ds}: {color}")

if __name__ == "__main__":
    metric = "NDCG@10"

    data = pd.read_csv("merged.csv", usecols=["data_set_name", "algorithm_name", metric, "num_interactions"])
    alg_list = data["algorithm_name"].unique().tolist()
    data = data.groupby(["data_set_name", "algorithm_name"])[metric].mean().to_frame().reset_index()

    # Filter only datasets in your list
    data = data[data["data_set_name"].isin(data_set_names)]

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

                x = x / max_total
                y = y / max_total

                axs[h, w].set(xlabel=alg1, ylabel=alg2)
                axs[h, w].set_xlim(0, 1)
                axs[h, w].set_ylim(0, 1)
                axs[h, w].set_aspect("equal", "box")
                axs[h, w].plot([0, 1], [0, 1])

                for _, row in merged.iterrows():
                    ds = row["data_set_name"]
                    x_val = row[f"{metric}_x"] / max_total
                    y_val = row[f"{metric}_y"] / max_total
                    color = dataset_colors.get(ds, "gray")
                    axs[h, w].plot(x_val, y_val, "o", markersize=4, color=color)

                w = (w + 1) % 7
                if w == 0:
                    h += 1

            plt.suptitle(f"{alg1}")
            pdf.savefig(fig)
            plt.close()
            print(f"Page done for {alg1}.")

        d = pdf.infodict()
        d['Title'] = 'Algorithm Performance Spaces'
